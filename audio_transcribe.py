"""
This script uses `faster-whisper` <https://github.com/SYSTRAN/faster-whisper>
 to transcribe an audio file to useful text files (txt, srt).
"""

# pyright: reportMissingTypeStubs=warning, reportUnknownMemberType=warning

from typing import List, Tuple
import argparse
from pathlib import Path
from datetime import timedelta
from faster_whisper import WhisperModel

def format_td(seconds:float, separator:str='.', digits:int=2):
    """format `seconds` using format `00:00:00.00` by default."""
    isec, fsec = divmod(round(seconds*10**digits), 10**digits)
    return f'{str(timedelta(seconds=isec)):0>8}{separator}{fsec:0{digits}.0f}'

def format_to_srt_ts(seconds:float):
    """format `seconds` to an SRT timestamp"""
    return format_td(seconds, ',', 3)

whisper_model_size = ['tiny', 'tiny.en',
                      'base', 'base.en',
                      'small', 'small.en', 'distil-small.en',
                      'medium', 'medium.en', 'distil-medium.en',
                      'large-v1', 'large-v2', 'large-v3', 'large',
                      'distil-large-v2', 'distil-large-v3']
whisper_device = ['cpu', 'cuda',
                  'auto']
whisper_computer_type = ['int8', 'int8_float32', 'int8_float16', 'int8_bfloat16',
                         'int16', 'float16', 'bfloat16', 'float32',
                         'default']

def transcribe(verbose:bool,
               model_size:str, device:str, compute_type:str,
               filepath:str, output_txt:bool, output_srt:bool):
    """transcribe the audio file at `filepath`"""
    # Run on GPU with FP16:
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # or run on GPU with INT8:
    # model = WhisperModel(model_size, device="cuda", compute_type="int8")
    # or run on CPU with INT8:
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(filepath, beam_size=5)

    if verbose:
        print(f"Detected language: '{info.language}'"
              f" with probability {info.language_probability:f}")

    transcribed_segments: List[Tuple[float, float, str]] = list()

    for segment in segments:
        if verbose:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        transcribed_segments.append((segment.start, segment.end, segment.text))

    if output_txt or output_srt:
        filename = Path(filepath).stem

        if output_txt:
            with open(f'{filename}.txt', 'w', encoding='utf-8') as txt_writer:
                txt_writer.writelines(s[2] for s in transcribed_segments)

        if output_srt:
            with open(f'{filename}.srt', 'w', encoding='utf-8') as srt_writer:
                groups = [(f"{counter}\n{format_to_srt_ts(s[0])} --> {format_to_srt_ts(s[1])}\n"
                           f"{s[2].strip()}\n\n")
                            for counter, s in enumerate(transcribed_segments, 1)]
                srt_writer.writelines(groups)

def main():
    """main()"""
    parser = argparse.ArgumentParser(prog='audio_transcribe',
                                     description='transcribe an audio file \
                                        to useful text files (txt, srt)',
                                     epilog='Happy transcribing!')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('--model_size', help='WhisperModel: model_size',
                        choices=whisper_model_size, default='small')
    parser.add_argument('--device', help='WhisperModel: device',
                        choices=whisper_device, default='auto')
    parser.add_argument('--compute_type', help='WhisperModel: compute_type',
                        choices=whisper_computer_type, default='default')
    parser.add_argument('filepath', help='path of the file to transcribe')
    parser.add_argument('-t', '--txt', help='output transcribed audio to a text file',
                        action='store_true')
    parser.add_argument('-s', '--srt', help='output transcribed audio to a SRT subtitle file',
                        action='store_true')
    args = parser.parse_args()
    transcribe(verbose=args.verbose,
               model_size=args.model_size, device=args.device, compute_type=args.compute_type,
               filepath=args.filepath, output_txt=args.txt, output_srt=args.srt)

if __name__ == "__main__":
    main()
