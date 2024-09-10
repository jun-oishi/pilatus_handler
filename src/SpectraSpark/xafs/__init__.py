#! /usr/bin/env python3

from .io import Xafs9809, read_ascii, merge_read
from .process import feffit, pair_feff, cif_feff
from .constants import FEFF_EXAFS_TMPL
from larch.xafs import xftf, path2chi, feffpath, autobk, feffit_transform, feffit_dataset

__all__ = ["Xafs9809", "read_ascii", "merge_read",
           "pair_feff", "cif_feff",
           "feffit",
           "FEFF_EXAFS_TMPL",
           "xftf", "path2chi", "feffpath", "autobk", "feffit_transform", "feffit_dataset"]

def main():
    import argparse

    parser_main = argparse.ArgumentParser(description="XAFS simulation and analysis tools")
    subparsers = parser_main.add_subparsers()

    parser_pair_feff = subparsers.add_parser("pair_feff", help="see `pair_feff -h`")
    parser_pair_feff.add_argument("abs", help="symbol of absorbing atom")
    parser_pair_feff.add_argument("scat", help="symbol of scattering atom")
    parser_pair_feff.add_argument("r", help="distance of 2 atoms", type=float)
    parser_pair_feff.add_argument("--folder", help="path to folder to save results", default="./feff")
    parser_pair_feff.add_argument("--edge", help="absorption edge", default="K")
    parser_pair_feff.add_argument("--kweight", help="k-weight for Fourier transform", type=int, default=3)
    parser_pair_feff.add_argument("--kmin", help="k-min of the window for Fourier transform", type=float, default=0)
    parser_pair_feff.add_argument("--kmax", help="k-max of the window for Fourier transform", type=float, default=20)
    parser_pair_feff.add_argument("--sig2", help="debye-waller factor", type=float, default=0.0002)
    parser_pair_feff.set_defaults(func=__pair_feff)

    args = parser_main.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser_main.print_help()


def __pair_feff(args):
    from ..util.io import savetxt
    import numpy as np

    path = pair_feff(abs=args.abs, scat=args.scat, r=args.r, folder=args.folder, edge=args.edge, sig2=args.sig2)
    path2chi(path)
    xftf(path, kmin=args.kmin, kmax=args.kmax, kweight=args.kweight)
    path.chir_pha = np.arctan2(path.chir_im, path.chir_re)
    header = f'{args.abs}-{args.scat} {args.r}A\n' \
             + 'r[A], chir_mag, chir_pha, chir_re, chir_im'
    data = np.array([path.r, path.chir_mag, path.chir_pha, path.chir_re, path.chir_im]).T
    dst = path.filename.replace('.dat', '_chir.dat')
    savetxt(dst, data, header=header, overwrite=True)
