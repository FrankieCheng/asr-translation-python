# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech V2 API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite_chirp2.py gcp_project_id
"""

# [START speech_transcribe_infinite_streaming_chirp2]

import argparse
import queue
import re
import sys
import time

from google.cloud import translate
from google.cloud.speech_v2 import SpeechClient
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
import pyaudio
import pprint
from google.oauth2 import service_account
import os

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

translate_client = translate.TranslationServiceClient()

def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def translate_text(project_id: str, location: str, text, tgt_lang):
    """Translating Text."""
    parent = f"projects/{project_id}/locations/{location}"
    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "target_language_code": tgt_lang,
        }
    )
    return response.translations[0].translated_text

def listen_print_loop(responses: object, stream: object, location:str, target_language: str, project_id: str) -> None:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Arg:
        responses: The responses returned from the API.
        stream: The audio stream to be processed.
    """
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        #print(transcript)
        translated_text = translate_text(project_id, location, transcript, target_language)
        #print(translated_text)


        result_seconds = 0
        result_micros = 0

        # Speech-to-text V2 result uses attribute result_end_offset instead of result_end_time
        # https://cloud.google.com/speech-to-text/v2/docs/reference/rest/v2/StreamingRecognitionResult
        if result.result_end_offset.seconds:
            result_seconds = result.result_end_offset.seconds

        if result.result_end_offset.microseconds:
            result_micros = result.result_end_offset.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + ": translation:" + translated_text + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + ": translation:" + translated_text + "\r")

            stream.last_transcript_was_final = False



def main(project_id: str, language: str, interim: bool, location:str, target_language: str) -> None:
    """start bidirectional streaming from microphone input to speech API"""
    #client = SpeechClient()
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com",
        )
    )

    recognition_config = cloud_speech_types.RecognitionConfig(
        explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
            sample_rate_hertz=SAMPLE_RATE,
            encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            audio_channel_count=1
        ),
        # auto_decoding_config=cloud_speech_types.AutoDetectDecodingConfig(),
        # language_codes=["de-DE"],
        # language_codes=["auto"],
        # translation_config=cloud_speech_types.TranslationConfig(target_language="fr-FR"),
        language_codes=[language],
        model="chirp_2",
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
            interim_results=interim
        )
    )

    config_request = cloud_speech_types.StreamingRecognizeRequest(
        #recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        #recognizer=f"projects/{project_id}/locations/{location}/recognizers/chirp-en-us-test",
        recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech_types.RecognitionConfig, audio: list) -> list:
        """Helper function to generate the requests list for the streaming API.

        Args:
            config: The speech recognition configuration.
            audio: The audio data.
        Returns:
            The list of requests for the streaming API.
        """
        yield config
        for chunk in audio:
            yield cloud_speech_types.StreamingRecognizeRequest(audio=chunk)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:
        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            audio_generator = stream.generator()

            # Transcribes the audio into text
            responses_iterator = client.streaming_recognize(
                requests=requests(config_request, audio_generator))

            listen_print_loop(responses_iterator, stream, location, target_language, project_id)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True

# credential=service_account.Credentials.from_service_account_file('credentials/demo.json',scopes=['https://www.googleapis.com/auth/cloud-platform'])
# credential_str = f"{os.getcwd()}/{credential}"

# GCP_CREDENTIALS_FILE = 'credentials/demo.json'
# credential = f"{os.getcwd()}/{GCP_CREDENTIALS_FILE}"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']=credential

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-project', action='store', dest='project_id', type=str, default='no-project',
		help='project')
    parser.add_argument('-language', action='store', dest='language', type=str, default='en-US',
		help='language')
    parser.add_argument('-interim', action='store', dest='interim', type=bool, default='True',
		help='interim')
    parser.add_argument('-location', action='store', dest='location', type=str, default='us-central1',
		help='location')
    parser.add_argument('-target_language', action='store', dest='target_language', type=str, default='zh-CN',
		help='target_language')
    args = parser.parse_args()
    main(args.project_id, args.language, args.interim, args.location, args.target_language)

# [END speech_transcribe_infinite_streaming_chirp2]
