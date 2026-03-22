#!/usr/bin/env python3

import argparse
import os
import re
import sys
import pmt

from PIL import Image
from gnuradio import blocks
from gnuradio import gr

import meteor_lrpt
import ccsds_image_decoder

from ccsds_image_assembler import CcsdsImageAssembler


FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<sps>\d+)SPS_(?P<freq>\d+)Hz\.cf32$"
)

CHANNEL_PORTS = {
    1: "msu_mr_1",
    4: "msu_mr_4",
}


def parse_input_filename(input_path):
    base_name = os.path.basename(input_path)
    match = FILENAME_RE.match(base_name)
    if match is None:
        raise ValueError(
            "Input filename must match "
            "YYYY-MM-DD_HH-MM-SS_<sample_rate>SPS_<frequency>Hz.cf32"
        )
    return {
        "date": match.group("date"),
        "time": match.group("time"),
        "sample_rate": int(match.group("sps")),
        "frequency_hz": int(match.group("freq")),
    }


def build_output_filename(input_path, channel_index):
    meta = parse_input_filename(input_path)
    directory = os.path.dirname(input_path)
    output_name = f"{meta['date']}_{meta['time']}_channel_{channel_index}.png"
    if directory == "":
        return output_name
    return os.path.join(directory, output_name)


class MessageNullSink(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="message_null_sink",
            in_sig=None,
            out_sig=None,
        )
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)

    def handle_msg(self, msg):
        pass


class MeteorChannelExtractor(gr.top_block):
    def __init__(self, input_path, sample_rate, channel_index):
        gr.top_block.__init__(self, "meteor_channel_extractor")

        if channel_index not in CHANNEL_PORTS:
            raise ValueError(
                f"Unsupported channel index: {channel_index}. "
                f"Supported values: {sorted(CHANNEL_PORTS.keys())}"
            )

        self.input_path = input_path
        self.sample_rate = int(sample_rate)
        self.channel_index = int(channel_index)
        self.channel_port = CHANNEL_PORTS[self.channel_index]

        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_gr_complex,
            self.input_path,
            False,
        )

        self.meteor_lrpt_0 = meteor_lrpt.meteor_lrpt(
            sample_rate=self.sample_rate
        )

        self.ccsds_image_decoder_0 = ccsds_image_decoder.CcsdsImageDecoder()
        self.ccsds_image_assembler_0 = CcsdsImageAssembler(width=1568)

        self.null_sink_constellation_0 = blocks.null_sink(gr.sizeof_gr_complex)
        self.null_sink_ber_0 = blocks.null_sink(gr.sizeof_float)
        self.null_sink_frequency_0 = blocks.null_sink(gr.sizeof_float)
        self.null_sink_snr_0 = blocks.null_sink(gr.sizeof_float)

        self.connect((self.blocks_file_source_0, 0), (self.meteor_lrpt_0, 0))

        self.connect((self.meteor_lrpt_0, 0), (self.null_sink_constellation_0, 0))
        self.connect((self.meteor_lrpt_0, 1), (self.null_sink_ber_0, 0))
        self.connect((self.meteor_lrpt_0, 2), (self.null_sink_frequency_0, 0))
        self.connect((self.meteor_lrpt_0, 3), (self.null_sink_snr_0, 0))

        self.msg_connect(
            (self.meteor_lrpt_0, self.channel_port),
            (self.ccsds_image_decoder_0, "in"),
        )

        self.msg_connect(
            (self.ccsds_image_decoder_0, "out"),
            (self.ccsds_image_assembler_0, "in"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Decode a selected Meteor M-N2 LRPT image channel from a .cf32 IQ recording."
    )
    parser.add_argument(
        "input_file",
        help="Input file in the format YYYY-MM-DD_HH-MM-SS_<sample_rate>SPS_<frequency>Hz.cf32",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Channel index to extract. Supported values: 1, 4. Default: 1",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default=None,
        help="Optional output PNG path. Default: <date>_<time>_channel_<index>.png",
    )
    args = parser.parse_args()

    try:
        meta = parse_input_filename(args.input_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.channel not in CHANNEL_PORTS:
        print(
            f"Unsupported channel index: {args.channel}. "
            f"Supported values: {sorted(CHANNEL_PORTS.keys())}",
            file=sys.stderr,
        )
        return 2

    output_file = args.output_file
    if output_file is None:
        output_file = build_output_filename(args.input_file, args.channel)

    tb = MeteorChannelExtractor(
        input_path=args.input_file,
        sample_rate=meta["sample_rate"],
        channel_index=args.channel,
    )

    print(f"Input file: {args.input_file}")
    print(f"Sample rate: {meta['sample_rate']}")
    print(f"Frequency: {meta['frequency_hz']} Hz")
    print(f"Channel: {args.channel}")
    print(f"Output file: {output_file}")

    tb.run()

    data = tb.ccsds_image_assembler_0.get_bytes()
    width, height = tb.ccsds_image_assembler_0.get_dimensions()

    if height <= 0:
        print("No complete image rows were assembled.", file=sys.stderr)
        return 1

    img = Image.frombytes("L", (width, height), data)
    img.save(output_file)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())