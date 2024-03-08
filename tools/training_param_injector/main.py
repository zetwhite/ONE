import argparse
import logging

from lib.circle_builder import CircleSOBuilder
from lib.traininfo_builder import TinfoSOBuilder
from lib.json_decodoer import decode as decode_from_json
from lib.json_encoder import encode as encode_to_json

TINFO_META_NAME = "CIRCLE_TRAINING"

def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle_parameter_injector',
        description='inject training parameter to the \'input_circle_file\'')
    parser.add_argument('json_file', type=str, help='json file that holds training parameter')
    parser.add_argument('input_circle_file', type=str, help='input circle file')
    parser.add_argument(
        'output_circle_file',
        type=str,
        nargs='?',
        help='output circle file with training parameter added\n'
        'if not given, input_circle_file is overwritten')
    parser.add_argument(
        '-v', '--verbosity', action="count", default=0, help='increase log verbosity')
    args = parser.parse_args()

    # if output_circle_file is not given, rewrite input_circle_file
    if not args.output_circle_file:
        args.output_circle_file = args.input_circle_file
    return args

def init_logger(verbosity: int):
    if verbosity >= 2:
        verbosity = 2
    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=log_level[verbosity])


if __name__ == "__main__":
    args = get_cmd_args()
    init_logger(args.verbosity)

    with open(args.json_file, 'r') as jsonfile:
        train_info = decode_from_json(jsonfile.read())

    logging.debug(f'{args.json_file} :')
    logging.debug(encode_to_json(train_info))
    
    tinfo_name = TINFO_META_NAME
    tinfo_buff = TinfoSOBuilder(train_info).get_buff()

    circle_builder = CircleSOBuilder(args.input_circle_file)
    circle_builder.inject_metadata(tinfo_name, tinfo_buff)
    circle_builder.export(args.output_circle_file)

    if check : 
        buf = circle_builder.get_metadata(TINFO_META_NAME)
        train_info = TinfoDeserializer(buf)
        print(encode_to_json(train_info))
