#! /usr/bin/env python3

import re
import os
import numpy as np
import warnings
from larch import Group
from larch import io
from larch.xrd.struct2xas import Struct2XAS
from larch.xafs import feffrunner, feffpath, FeffPathGroup, feffit_report
from xraydb import atomic_number

from SpectraSpark.util.basic_calculation import nm2ev
from .constants import FEFF_EXAFS_TMPL

FEFF_EXECUTABLE = os.environ.get("FEFF_EXECUTABLE", "feff8l")

class Xafs9809:
    def __init__(self, src, ch_fluor=-1, preferred_ch='fluor', angle='observe'):
        with open(src, "r") as f:
            lines = f.readlines()

        self.beamline = lines[0].strip()
        self.sample = lines[1].split()[0]
        self.ring = lines[3].strip()
        self.mono = lines[4].strip()
        if _match := re.search(r"D=( *)([0-9\.]+)( *)A", self.mono):
            self.dspacing = float(_match.group(2))
        else:
            raise ValueError(f"cannot find d-spacing in mono: {src}")
        self.param = [lines[5].strip(), lines[6].strip()]

        n_blocks = 0
        blocks = []
        for ln in range(9, len(lines)):
            pattern = r"^" + r" +".join([r"([0-9\.]+)"] * 6) + r"$"
            if re.search(pattern, lines[ln].strip()):
                n_blocks += 1
                ln += 1
                blocks.append(lines[ln].strip())
            else:
                break
        self.n_blocks = n_blocks
        self.blocks = tuple(blocks)

        data = np.loadtxt(src, skiprows=13+ln)
        self.angle_control = data[:, 0]
        self.angle_observe = data[:, 1]
        if angle.lower() in ('control', 'c'):
            self.energy = nm2ev(1e-1 * 2 * self.dspacing \
                                * np.sin(np.radians(self.angle_control)))
        elif angle.lower() in ('observe', 'o'):
            self.energy = nm2ev(1e-1 * 2 * self.dspacing \
                                * np.sin(np.radians(self.angle_observe)))
        else:
            raise ValueError(f"invalid angle: {angle}")
        self.time = data[:, 2]
        self.i0 = data[:, 3]
        self.trans = data[:, 4]
        if ch_fluor > 0:
            self.fluor = data[:, ch_fluor+2]

        if preferred_ch == 'fluor' and ch_fluor > 0:
            self.mu = self.fluor / self.i0
        else:
            self.mu = -np.log(self.trans / self.i0)

        return

    def write_ascii(self, dst, columns=('mu')):
        data = self.energy.reshape(-1, 1)
        fmt = ['%7.f']
        for col in columns:
            if col == 'energy':
                data = np.hstack((data, self.energy.reshape(-1, 1)))
                fmt.append('%.4f')
            elif col == 'time':
                data = np.hstack((data, self.time.reshape(-1, 1)))
                fmt.append('%.2f')
            elif col == 'mu':
                data = np.hstack((data, self.mu.reshape(-1, 1)))
                fmt.append('%7.f')
            elif col == 'i0':
                data = np.hstack((data, self.i0.reshape(-1, 1)))
                fmt.append('%i')
            elif col == 'trans':
                data = np.hstack((data, self.trans.reshape(-1, 1)))
                fmt.append('%i')
            elif col == 'mutrans':
                mutrans = -np.log(self.trans / self.i0)
                data = np.hstack((data, mutrans.reshape(-1, 1)))
                fmt.append('%7.f')
            elif col == 'fluor':
                data = np.hstack((data, self.fluor.reshape(-1, 1)))
                fmt.append('%i')
            elif col == 'mufluor':
                mufluor = -np.log(self.fluor / self.i0)
                data = np.hstack((data, mufluor.reshape(-1, 1)))
                fmt.append('%7.f')
            else:
                raise ValueError(f"invalid column: {col}")
        header = " ".join(columns)
        np.savetxt(dst, data, header=header, fmt=fmt)
        return

    def as_group(self):
        g = Group(energy=self.energy, mu=self.mu)
        return g

    @classmethod
    def to_ascii(cls, src, dst, columns=('mu')):
        data = cls(src)
        data.write_ascii(dst, columns=columns)
        return

def read_ascii(src, *, labels=[], skiprows=-1)->Group:
    if skiprows == 0:
        # skiprowsが0の場合は1列目がenergy, 2列目がmuとして読み込む
        return io.read_ascii(src, labels=['energy', 'mu'])

    if skiprows < 0:
        # #から始まる行はheaderとして無視する
        skiprows = 0
        with open(src, "r") as f:
            for ln, line in enumerate(f):
                if line.startswith("#"):
                    skiprows += 1
                else:
                    break

    if len(labels) == 0:
        # labelsが指定されていない場合はheaderの最後の行をlabelsとして使う
        with open(src, "r") as f:
            header = f.readlines()[skiprows-1]
        header = header[1:]   # remove '#'
        labels = header.strip().split()

    data = io.read_ascii(src, labels=labels)
    if hasattr(data, 'e') and not hasattr(data, 'energy'):
        data.energy = data.e
    if hasattr(data, 'xmu') and not hasattr(data, 'mu'):
        data.mu = data.xmu
    return data

def merge(groups):
    """muを持つGroupのリストを結合する"""
    energy = groups[0].energy
    mu = np.empty((len(groups), len(energy)), dtype=float)
    for i, group in enumerate(groups):
        if len(energy) != len(group.energy):
            raise ValueError("energy length mismatch")
        elif not np.allclose(energy, group.energy):
            raise ValueError("energy mismatch")
        elif not hasattr(group, "mu") or group.mu.size != len(energy):
            raise ValueError("mu mismatch")
        mu[i] = group.mu
    mu_mean = np.nanmean(mu, axis=0)
    merged = Group(energy=energy, mu=mu_mean)
    return merged

def merge_read(files, *, formats='xafs9809', **kwargs):
    if formats.lower() == 'xafs9809':
        data = [Xafs9809(file, angle='control', **kwargs).as_group() for file in files]
    elif formats.lower() == 'ascii':
        data = [read_ascii(file, **kwargs) for file in files]
    else:
        raise ValueError(f"invalid format: {formats}")
    return merge(data)

def pair2feffinp(abs, scat, r, *, folder='./feff', title='', edge='K',
                        sig2=None, temperature=300, debye_temperature=None):
    """吸収原子と散乱原子2原子のみのfeff.inpを生成してそのフォルダを返す"""
    z_abs, z_scat = atomic_number(abs), atomic_number(scat)
    title = f"{abs}-{scat}_reff{r:.4f}" if title == '' else title
    outdir = f"{folder}/{title}"
    if os.path.isfile(outdir):
        raise ValueError(f"{outdir} already exists")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    src = open(FEFF_EXAFS_TMPL, "r")
    dst = open(f'{folder}/{title}/feff.inp', "w")
    for line in src:
        if line.startswith(r"{title}"):
            line = f"TITLE {title} reff={r:.4f}\n"
        elif r"{edge}" in line:
            line = line.replace(r"{edge}", edge)
        elif r"{radius}" in line:
            line = line.replace(r"{radius}", f"{r*1.1:.3f}")
        elif r"SIG2" in line:
            if sig2 is not None:
                line = line.replace(r"{use_sig2} ", "") \
                           .replace(r"{sig2}", f"{sig2:.10f}")
            else:
                line = line.replace(r"{use_sig2} ", "* ")
        elif r"DEBYE" in line:
            if debye_temperature is not None:
                line = line.replace(r"{use_debye} ", "") \
                           .replace(r"{temperature}", f"{temperature:.1f}") \
                           .replace(r"{debye_temperature}", f"{debye_temperature:.1f}")
            else:
                line = line.replace(r"{use_debye} ", "* ")
        elif line.startswith(r"{potentials}"):
            header =  "* ipot  Z   tag"
            lines = [f"     0  {z_abs}  {abs}",
                     f"     1  {z_scat} {scat}"]
            line = header + "\n" + "\n".join(lines)
        elif line.startswith(r"{atoms}"):
            header = "* x    y    z   ipot  tag   distance   occupancy"
            lines = [f"  0.00000  0.00000  0.00000  0  {abs}  0.00000   *1",
                     f"  0.00000  0.00000  {r:.5f}  1  {scat}  {r:.5f}   *1"]
            line = header + "\n" + "\n".join(lines)

        dst.write(line)

    src.close()
    dst.close()

    return outdir

def cif2feffinp(cif, abs_atom, radius=7.0, *, folder='./feff', abs_site=-1)->str:
    """cifファイルからfeff.inpを生成してそのフォルダを返す"""
    struct = Struct2XAS(cif, abs_atom=abs_atom)
    n_sites = len(struct.get_abs_sites())
    if abs_site==-1 and n_sites>1:
        raise ValueError(f"abs_site must be specified for {n_sites} sites")
    elif abs_site==-1:
        abs_site = 0
    struct.set_abs_site(abs_site)

    struct.make_input_feff(radius=radius, template=FEFF_EXAFS_TMPL,
                           parent_path=folder)
    return struct.outdir

def run_feff(outdir, feffinp='feff.inp')->list[FeffPathGroup]:
    runner = feffrunner(folder=outdir, feffinp=feffinp, verbose=False)
    runner.run(exe=FEFF_EXECUTABLE)

    paths = [f for f in os.listdir(outdir) if re.search(r"^feff\d{4}\.dat$", f)]
    paths.sort()
    ret = []
    for f in paths:
        path = feffpath(os.path.join(outdir,f))
        label_path(path)
        ret.append(path)

    return ret

def label_path(path, inplace=True):
    geom = 'abs-' + '-'.join([item[0] for item in path.geom[1:]]) + f'x{path.degen:.1f}'
    if inplace:
        path.label = geom
    return geom

def save_rbkg(group:Group, dst:str='', *, fmt='%.8f', only_mu=False)->str:
    """groupからrbkgのパラメタと結果を抽出して保存する"""
    if not hasattr(group, 'autobk_details'):
        if only_mu:
            warnings.warn("group does not seem autobk-processed")
        else:
            raise ValueError("group does not seem autobk-processed")

    if dst == '':
        dst = group.filename.split('.')[0] + '_bkg.dat'
    elif dst.endswith('/'):
        dst += group.filename.split('.')[0] + '_bkg.dat'

    if only_mu:
        table = np.array([group.energy, group.mu]).T
        header = 'energy\tmu'
        np.savetxt(dst, table, header=header, fmt=fmt, delimiter='\t')
        return dst

    autobk_details = group.autobk_details
    keys = ('ek0', 'iek0', 'iemax', 'irbkg', 'kmax', 'kmin')
    details = {key: getattr(autobk_details, key) for key in keys if not key.startswith('_')}
    details['rbkg'] = group.rbkg
    details['e0'] = group.e0
    details['src'] = group.filename
    column_labels = ['energy', 'mu', 'pre_edge', 'post_edge', 'bkg', 'chie']
    table = np.array([getattr(group, label) for label in column_labels]).T
    header = '\n'.join([f"{key}: {details[key]}" for key in details])
    header += '\n' + '\t'.join(column_labels)
    np.savetxt(dst, table, header=header, fmt=fmt, delimiter='\t')
    return dst

def save_chik(group:Group, dst:str='', *, fmt='%.8f')->str:
    """groupからchikを抽出して保存する"""
    if not hasattr(group, 'k'):
        raise ValueError("group does not seem k-space processed")

    if dst == '':
        dst = group.filename.split('.')[0] + '_chik.dat'
    elif dst.endswith('/'):
        dst += group.filename.split('.')[0] + '_chik.dat'

    headers = f'src: {group.filename}\n' \
              + f'e0: {group.e0}\n' \
              + f'k\tchik'
    table = np.array([group.k, group.chi]).T
    np.savetxt(dst, table, header=headers, fmt=fmt)
    return dst

def save_chir(group:Group, dst:str='', *, fmt='%.8f')->str:
    """groupからchirを抽出して保存する"""
    if not hasattr(group, 'r'):
        raise ValueError("group does not seem r-space processed")

    if dst == '':
        dst = group.filename.split('.')[0] + '_chir.dat'
    elif dst.endswith('/'):
        dst += group.filename.split('.')[0] + '_chir.dat'

    headers = f'src: {group.filename}\n' \
              + f'e0: {group.e0}\n' \
              + f'r\tchir_mag\tchir_pha\tchir_re\tchir_im'
    table = np.array([group.r, group.chir_mag, group.chir_re, group.chir_im]).T
    np.savetxt(dst, table, header=headers, fmt=fmt)
    return dst

def save_feffit(out:Group, dst:str='', *, fmt='%.8f',
                with_rbkg=True, with_chik=True)->list[str]|str:
    """feffitの結果を保存する"""
    if not out.success:
        warnings.warn("feffit failed")

    simultaneous = len(out.datasets) > 1
    if not simultaneous:
        if dst == '':
            dst = out.datasets[0].data.filename.split('.')[0] + '_feffit.dat'
        elif dst.endswith('/'):
            dst += out.datasets[0].data.filename.split('.')[0] + '_feffit.dat'
    elif dst == '' or dst.endswith('/'):
        raise ValueError("dst must be specified for simultaneous fitting")

    if not dst.endswith('.dat'):
        dst = dst + '.dat'

    saved = []

    report = feffit_report(out)

    with open(dst.replace('.dat', '.log'), 'w') as f:
        f.write(report)


    for dataset in out.datasets:
        filename = dataset.data.filename
        label = filename.split('.')[0]
        _dst = dst
        if simultaneous:
            _dst = dst.replace('.dat', f'_{label}.dat')
        model, data = dataset.model, dataset.data
        if with_rbkg:
            save_rbkg(data, dst=_dst.replace('.dat', '_rbkg.dat'), fmt=fmt, only_mu=False)
        if with_chik:
            save_chik(data, dst=_dst.replace('.dat', '_chik.dat'), fmt=fmt)
        header = 'r/A chi_exp/A^-4 chi_fit/A^-4'
        r, chi_exp, chi_fit = model.r, data.chir_mag, model.chir_mag
        np.savetxt(_dst, np.array([r, chi_exp, chi_fit]).T, header=header, fmt=fmt)
        saved.append(_dst)

    if simultaneous:
        return saved
    else:
        return saved[0]

def save(group:Group, dst:str='./', feffit_out:Group=None):
    if not hasattr(group, 'autobk_details'):
        warnings.warn("group does not seem autobk-processed")
        return
    _dst = save_rbkg(group, dst=dst)
    print(f'{_dst} saved')

    if not hasattr(group, 'k'):
        return
    _dst = save_chik(group, dst=dst)
    print(f'{_dst} saved')

    if not hasattr(group, 'r'):
        return
    else:
        _dst = save_chir(group, dst=dst)
        print(f'{_dst} saved')

    if feffit_out is None:
        return
    _dst = save_feffit(feffit_out, dst=dst)
    print(f'{_dst} saved')