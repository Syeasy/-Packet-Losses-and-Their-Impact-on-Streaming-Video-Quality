#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     dfr_simulation.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-05-15
#           2023-04-11
#           2024-04-25
#
# @brief Skeleton code for the simulation of video streaming to investigate the
#        impact of packet losses on the quality of video streaming based on
#        decodable frame rate (DFR)
#


import argparse
import math
import numpy as np
import os
import sys
import sgm_generate

sys.path.insert(0, os.getcwd())  # assume that your own modules in the current directory
from conv_interleave import conv_interleave, conv_deinterleave
from sgm_generate import sgm_generate


def dfr_simulation_v2(
        random_seed,
        num_frames,
        loss_probability,
        video_trace,
        fec,
        ci):
    np.random.seed(random_seed)  # set random seed

    # N.B.: Obtain the information of the whole frames to create a loss
    # sequence in advance due to the optional convolutional
    # interleaving/deinterleaving.
    with open(video_trace, "r") as f:
        lines = f.readlines()[1:num_frames + 1]  # the first line is a comment.

    f_number = np.empty(num_frames, dtype=np.uint)  # 统计所有
    f_type = [''] * num_frames
    f_pkts = np.empty(num_frames, dtype=np.uint)  # the number of packets per frame
    for i in range(num_frames):
        f_info = lines[i].split()
        f_number[i] = int(f_info[0])  # str -> int number index
        f_type[i] = f_info[2]  # 类型
        f_pkts[i] = math.ceil(int(f_info[3]) / (188 * 8))  # split packets per frame

    # symbol loss sequence
    p = 1e-4
    q = p * (1.0 - loss_probability) / loss_probability
    n_pkts = sum(f_pkts)  # the number of packets for the whole frames
    pkt_size = 204 if fec else 188
    if ci:
        # apply convolutional interleaving/deinterleaving.
        # N.B.:
        # 1. Append 2244 zeros before interleaving.
        # 2. Interleaved sequence experiences symbol losses.
        # 3. Remove leading 2244 elements after deinterleaving.
        # TODO: Implement.
        # assume origin sequence is all 1
        d1 = [int(i) for i in "17,34,51,68,85,102,119,136,153,170,187".split(',')]
        d2 = d1[::-1]
        len_sending_seq = pkt_size * n_pkts
        sending_seq = conv_interleave(np.concatenate([np.ones(len_sending_seq), np.zeros(2244)]), d1)
        loss_seq = sgm_generate(random_seed, len_sending_seq + 2244, p, q) ^ 1
        received_seq = conv_deinterleave(loss_seq * sending_seq, d2)[2244:]

    else:
        # TODO: Implement.
        len_sending_seq = pkt_size * n_pkts
        sending_seq = np.ones(len_sending_seq)
        loss_seq = sgm_generate(random_seed, len_sending_seq, p, q) ^ 1
        received_seq = loss_seq * sending_seq
    # initialize variables.
    idx = -1
    for j in range(2):
        idx = f_type.index('I', idx + 1)  # Counting the index of the second 'I' and subtracting the first index (which
        # is necessarily 1) from this index gives the GOP length
    gop_size = f_number[idx]  # N.B.: the frame number of the 2nd I frame is GOP size.
    num_b_frames = f_number[1] - f_number[0] - 1  # between I and the 1st P frames
    num_pkts_received = 0
    num_frames_decoded = 0
    num_frames_received = 0
    losses = np.zeros(n_pkts)
    frame_loss = np.zeros(num_frames, dtype=bool)
    frame_decoded = np.zeros(num_frames, dtype=bool)
    # main loop
    for i in range(num_frames):
        # frame loss
        if fec:
            # TODO: Set "frame_loss" based on "pkt_losses" with FEC.
            received_frame = received_seq[num_pkts_received * pkt_size: (num_pkts_received + f_pkts[i]) * pkt_size]
            pkt_cnts = received_frame.reshape(-1, pkt_size * 1)

            for pkt_idx in range(len(pkt_cnts)):
                num_sym_err = np.count_nonzero(pkt_cnts[pkt_idx] == 0)
                if num_sym_err > 8 * 1:
                    losses[num_pkts_received + pkt_idx] += 1

        else:
            # TODO: Set "frame_loss" based on "pkt_losses" without FEC.
            received_frame = received_seq[num_pkts_received * pkt_size: (num_pkts_received + f_pkts[i]) * pkt_size]
            pkt_cnts = received_frame.reshape(-1, pkt_size * 1)
            for pkt_idx in range(len(pkt_cnts)):
                if np.count_nonzero(pkt_cnts[pkt_idx] == 0) > 0 * 1:
                    losses[num_pkts_received + pkt_idx] += 1
        pkt_losses = sum(losses[num_pkts_received:num_pkts_received + f_pkts[i]])
        num_pkts_received += f_pkts[i]
        num_frames_received += 1
        if pkt_losses == 0:
            frame_loss[i] = False
        else:
            frame_loss[i] = True

        # frame decodability
        if not frame_loss[i]:  # see the fec-dependent handling of "frame_loss" above.
            match f_type[i]:
                case 'I':
                    # TODO: Implement.
                    frame_decoded[i] = True
                    num_frames_decoded += 1
                case 'P':
                    # TODO: Implement.
                    floor_idx = np.where(f_number == f_number[i] - num_b_frames - 1)[0][0]
                    if frame_decoded[floor_idx]:
                        frame_decoded[i] = True
                        num_frames_decoded += 1
                case 'B':
                    # TODO: Implement.
                    ceil_idx = np.where(f_number == math.ceil(f_number[i] / (num_b_frames + 1)) * (num_b_frames + 1))[0]
                    floor_idx = np.where(f_number == math.floor(f_number[i] / (num_b_frames + 1)) * (num_b_frames + 1))[
                        0]
                    if frame_decoded[floor_idx] and frame_decoded[ceil_idx]:
                        frame_decoded[i] = True
                        num_frames_decoded += 1
                case _:
                    sys.exit("Unkown frame type is detected.")
    return num_frames_decoded / num_frames  # DFR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--num_frames",
        help="number of frames to simulate; default is 10000",
        default=10000,
        type=int)
    parser.add_argument(
        "-P",
        "--loss_probability",
        help="overall loss probability; default is 0.1",
        default=0.1,
        type=float)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="seed for numpy random number generation; default is 777",
        default=777,
        type=int)
    parser.add_argument(
        "-V",
        "--video_trace",
        help="video trace file; default is 'silenceOfTheLambs_verbose'",
        default="silenceOfTheLambs_verbose",
        type=str)

    # convolutional interleaving/deinterleaving (CI); default is False
    parser.add_argument('--ci', dest='ci', action='store_true')
    parser.add_argument('--no-ci', dest='ci', action='store_false')
    parser.set_defaults(ci=False)

    # forward error correction (FEC); default is False (i.e., not using FEC)
    parser.add_argument('--fec', dest='fec', action='store_true')
    parser.add_argument('--no-fec', dest='fec', action='store_false')
    parser.set_defaults(fec=False)

    args = parser.parse_args()

    # set variables using command-line arguments
    num_frames = args.num_frames
    # loss_model = args.loss_model
    loss_probability = args.loss_probability
    video_trace = args.video_trace
    ci = args.ci
    fec = args.fec
    # trace = args.trace

    # run simulation and display the resulting DFR.
    dfr = dfr_simulation_v2(
        args.random_seed,
        args.num_frames,
        args.loss_probability,
        args.video_trace,
        args.fec,
        args.ci)
    print(f"Decodable frame rate = {dfr:.2%}\n")
