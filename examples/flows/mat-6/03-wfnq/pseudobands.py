#!/usr/bin/env python
import h5py
import numpy as np
import scipy as sp
from scipy import signal
import logging
import time
import sys
# from optimize_funs import optimize, alpha, w
import numpy as np
from scipy.optimize import minimize_scalar, fsolve, Bounds
import warnings

# Constants based on 3D free electron gas density of states
C = 1 / (12 * np.pi**8) # atomic units
B = np.sqrt(2) / np.pi**2

def alpha(beta, E0, Emax, nspbps, nslice): 
    dE = Emax - E0
    return (dE * (np.exp(beta)-1)) / (np.exp(beta) * (np.exp(nslice * beta) - 1))


def w(beta, j, E0, Emax, nspbps, nslice): 
    return alpha(beta, E0, Emax, nspbps, nslice) * np.exp(j * beta)

def Ebar(beta, j, E0, Emax, nspbps, nslice): 
    a = alpha(beta, E0, Emax, nspbps, nslice)
    width = w(beta, j, E0, Emax, nspbps, nslice)
    return E0 + a * np.exp(beta) * (np.exp(j * beta) - 1) / (np.exp(beta) - 1) - width / 2


def Loss(beta, E0=1, Emax=10, nspbps=1, nslice=10):
    out = 0
    for j in range(1, nslice+1):
        Ei = Ebar(beta, j, E0, Emax, nspbps, nslice)
        wi = w(beta, j, E0, Emax, nspbps, nslice)

        assert Ei > 0, Ei
        assert wi > 0, wi
        
        # FEG approximation to dimension of the subspace breaks down, let dim = 1 in this case
        if B * wi * np.sqrt(Ei) < 1:
            out += C * wi**2 / Ei**2
            continue
            
        l = C * wi**2 / Ei**2 + 1 / (Ei**2 * nspbps) * (1 - 1 / (B * wi * np.sqrt(Ei)))
        out += l
        
    return out


def optimize(E0=1, Emax=10, nspbps=1, nslice=10):
    
    dE = Emax - E0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        min_dim = fsolve(lambda x: B * x * np.sqrt(E0 + x/2) - 1, 4)
        low = max(fsolve(lambda x: alpha(x, E0, Emax, nspbps, nslice)*np.exp(x) 
                     - max(min_dim), .2*np.ones(3), maxfev=1000))
        high = max(fsolve(lambda x: B * w(x, nslice, E0, Emax, nspbps, nslice) 
                      * np.sqrt(Ebar(x, nslice, E0, Emax, nspbps, nslice)) - 1,
                      .2*np.ones(3), maxfev=1000))
   
        tol=1e-9
        bnds = (tol, 1-tol)
        
        result = minimize_scalar(Loss, args=(E0, Emax, nspbps, nslice), bounds=bnds, method='bounded')
    
    print(f'minimization results: \n{result}')
    print(f'alpha: {alpha(result.x, E0, Emax, nspbps, nslice)}')
    
    return result
    



# so that logger doesn't try to truncate arrays
np.set_printoptions(threshold=sys.maxsize)


def construct_blocks(el, n_copy, nslice, nspbps, uniform_width, max_freq):
    if uniform_width is None or uniform_width == 0.0:
        assert max_freq == 0.0

    # optimizing parameters
    if max_freq > 0:
        e0 = el[n_copy]
        slices = uniform_width * np.arange(max_freq // uniform_width - 1) + e0
        start_exp = len(slices)

        e0new = slices[-1] + uniform_width
        emax = el[-1]
        assert e0 > 0
        assert emax > e0
        assert emax > e0new
        assert e0new > e0

        res = optimize(e0new, emax, nspbps, nslice)
        beta = res.x

        widths = np.insert(np.cumsum([w(beta, j, e0new, emax, nspbps, nslice) for j in range(1, nslice + 1)]), 0,
                           0) + e0new

        slices = np.concatenate((slices, widths))

    else:
        e0 = el[n_copy]
        emax = el[-1]

        assert e0 > 0
        assert emax > e0

        res = optimize(e0, emax, nspbps, nslice)
        beta = res.x

        slices = np.insert(np.cumsum([w(beta, j, e0, emax, nspbps, nslice) for j in range(1, nslice + 1)]), 0, 0) + e0

        start_exp = 0

    assert slices[0] == e0
    slices[-1] = emax

    blocks = []
    nb_out = n_copy
    si = 0

    while si <= len(slices) - 2:
        if len(blocks) > 1:
            l = len(blocks)
            assert blocks[l - 1][0] == blocks[l - 2][1] + 1

        first_en = slices[si]
        assert first_en > 0

        last_en = slices[si + 1]
        assert last_en >= first_en
        si += 1

        first_idx = list(np.where(el >= first_en))
        last_idx = list(np.where(el <= last_en))

        # no bands in slice, mostly an issue for valence states (usually sparse)
        if first_idx[0][0] == last_idx[0][-1]:
            blocks.append([first_idx[0][0], last_idx[0][-1]])
            continue
        elif first_idx[0][0] > last_idx[0][-1]:
            continue

        blocks.append([first_idx[0][0], last_idx[0][-1]])

        nb_out += 1

    return np.asarray(blocks), start_exp


# fix signs/relative alignment
def fix_blocks(blocks, vc, ifmax, nb_orig):
    if vc == 'v':
        np.flip(blocks)
        blocks *= -1
        blocks += ifmax - 1
        assert blocks[-1][-1] == 0

    elif vc == 'c':
        blocks += ifmax
        assert blocks[-1][-1] == nb_orig - 1


# Sanity checks, block construction for WFN and WFNq simultaneously
def check_and_block(fname_in=None, fname_in_q=None, nv=-1, nc=100, nslice_v=10, nslice_c=100, uniform_width=None,
                    max_freq=0., nspbps_v=2, nspbps_c=2, verbosity=0, **kwargs):
    nv = int(nv)
    nc = int(nc)

    if uniform_width is None:
        assert max_freq == 0.
    else:
        assert uniform_width >= 0

    assert nv >= -1
    assert nc >= -1
    assert nslice_v > 0
    assert nslice_c > 0

    assert not (nv == -1 and nc == -1), 'Just copying input WFN file, no need to use this script for that.'

    if nv != -1:  # not just copying
        assert nspbps_v >= 2, 'You must use at least 2 bands per slice to obtain meaningful results!'
    if nc != -1:  # not just copying
        assert nspbps_c >= 2, 'You must use at least 2 bands per slice to obtain meaningful results!'

    f_in = h5py.File(fname_in, 'r')
    f_in_q = h5py.File(fname_in_q, 'r')

    assert f_in['mf_header/flavor'][()] == 2, 'Complex WFN file required for use of pseudobands!'
    assert f_in_q['mf_header/flavor'][()] == 2, 'Complex WFNq file required for use of pseudobands!'

    en_orig = f_in['mf_header/kpoints/el'][()]
    nb_orig = f_in['mf_header/kpoints/mnband'][()]
    ifmax = np.asarray(f_in['mf_header/kpoints/ifmax'][()], dtype=int)[0]  # spin 1

    en_orig_q = f_in_q['mf_header/kpoints/el'][()]
    ifmax_q = np.asarray(f_in_q['mf_header/kpoints/ifmax'][()], dtype=int)[0]  # spin 1
    # TODO: add spin support

    try:
        assert np.array_equal(ifmax / ifmax[0], np.ones(ifmax.shape[0]))
    except AssertionError as err:
        logger.error(
            '''ifmax is not the same across all kpoints. 
            This script does not currently support non-uniform band occupations.''')
        raise err

    try:
        assert np.array_equal(ifmax_q / ifmax_q[0], np.ones(ifmax_q.shape[0]))
    except AssertionError as err:
        logger.error(
            '''ifmax_q is not the same across all kpoints. 
            This script does not currently support non-uniform band occupations.''')
        raise err

    try:
        assert ifmax[0] == ifmax_q[0]
    except AssertionError as err:
        logger.error(
            'ifmax is different from ifmax_q. This is nonphysical, make sure your WFN and WFNq files are correct.')
        raise err

    ifmax = ifmax[0]  # assumes constant ifmax == nonmetal

    logger.info(f'ifmax = {ifmax}')

    lastcopy_en_v = en_orig[:, :, ifmax - 1 - nv]
    lastcopy_en_c = en_orig[:, :, ifmax - 1 + nc]

    if any(np.in1d(lastcopy_en_v, en_orig[..., ifmax - nv:ifmax])):
        logger.error('Chosen nv cuts degenerate bands, use different nv. Try degeneracy_check.x')
        raise Exception('Chosen nv cuts degenerate bands, use different nv. Try degeneracy_check.x')
    if any(np.in1d(lastcopy_en_c, en_orig[..., ifmax:ifmax + nc - 2])):
        logger.error('Chosen nc cuts degenerate bands, use different nc. Try degeneracy_check.x')
        raise Exception('Chosen nc cuts degenerate bands, use different nc. Try degeneracy_check.x')

    vbmcbm = np.concatenate((en_orig[..., ifmax - 1:ifmax + 1], en_orig_q[..., ifmax - 1:ifmax + 1]), axis=1)

    # ifmax is 1-indexed, en_orig is 0-indexed
    # also fortran order...
    fermi = np.mean([np.max(vbmcbm[..., 0]), np.min(vbmcbm[..., 1])])
    logger.info(f'E_Fermi = {fermi} Ry')

    assert nv <= ifmax
    assert nc <= nb_orig - ifmax

    en_orig -= fermi
    en_orig_q -= fermi

    # average over kpoints
    el_all_v = -np.flip(np.mean(np.concatenate((en_orig[..., :ifmax], en_orig_q[..., :ifmax]), axis=1)[0], axis=0))

    el_c = np.mean(en_orig[0, :, ifmax:None], axis=0)

    if nv != -1:
        blocks_v, start_exp_v = construct_blocks(el_all_v, nv, nslice_v, nspbps_v, uniform_width, max_freq)
        fix_blocks(blocks_v, 'v', ifmax, nb_orig)
        blocks_en_v = [[np.mean(en_orig[..., b[0]]), np.mean(en_orig[..., b[1]])] for b in blocks_v]

        if verbosity > 0:
            logger.info(f'index of first exponential valence slice: {start_exp_v}')
            logger.info(f'valence slices: {blocks_v}')
            logger.info(f'valence slice energies (Ry) (relative to E_Fermi): {blocks_en_v}')

        nslices_v = len(blocks_v)
        nspb_v = nslices_v * nspbps_v
        nb_out_v = nv + nspb_v

        try:
            assert nb_out_v <= ifmax
        except AssertionError as err:
            logger.error(
                '''More total valence states (copied + SPBs) than original valence bands ==> no computational savings 
                in GW steps. Choose smaller nv, nspbps_v, or larger efrac_v''')
            raise err

        assert len(blocks_v) == nslices_v

    else:
        logger.info('No valence pseudobands')

    if nc != -1:
        blocks_c, start_exp_c = construct_blocks(el_c, nc, nslice_c, nspbps_c, uniform_width, max_freq)
        fix_blocks(blocks_c, 'c', ifmax, nb_orig)
        blocks_en_c = [[np.mean(en_orig[..., b[0]]), np.mean(en_orig[..., b[1]])] for b in blocks_c]

        if verbosity > 0:
            logger.info(f'index of first exponential conduction slice: {start_exp_c}')
            logger.info(f'conduction slices: {blocks_c}')
            logger.info(f'conduction slice energies (Ry) (relative to E_Fermi): {blocks_en_c}')

        nslices_c = len(blocks_c)
        nspb_c = nslices_c * nspbps_c
        nb_out_c = nc + nspb_c

        assert len(blocks_c) == nslices_c

    else:
        logger.info('No conduction pseudobands')

    # get SPB params, then close files
    params_from_parabands = {}

    try:
        params = f_in['parabands/pseudobands']

        params_from_parabands['nc'] = params['nc'][()]
        params_from_parabands['n_subspaces'] = params['n_subspaces'][()]
        params_from_parabands['num_per_subspace'] = params['num_per_subspace'][()]
    except:
        pass

    f_in.close()
    f_in_q.close()

    if nv != -1 and nc != -1:
        logger.info(
            f'''nslices_c = {nslices_c}\n nslices_v = {nslices_v}\n nspb_c_total = {nspb_c}\n 
            nspb_v_total = {nspb_v}\n nb_out_c_total = {nb_out_c}\n nb_out_v_total = {nb_out_v}\n''')
        return blocks_v, blocks_c, ifmax, params_from_parabands
    elif nv == -1 and nc != -1:
        logger.info(f'''nslices_c = {nslices_c}\n nspb_c_total = {nspb_c}\n nb_out_c_total = {nb_out_c}\n''')
        return None, blocks_c, ifmax, params_from_parabands
    elif nv != -1 and nc == -1:
        logger.info(f'''nslices_v = {nslices_v}\n nspb_v_total = {nspb_v}\n nb_out_v_total = {nb_out_v}\n''')
        return blocks_v, None, ifmax, params_from_parabands


def fill_pseudoband_params(fout, vc, nprot, nslice, nspbps, max_freq, uniform_width):
    group = 'pseudobands/' + vc

    fout[group].create_dataset('nprot', (), data=nprot)
    fout[group].create_dataset('nslice', (), data=nslice)
    fout[group].create_dataset('nspbps', (), data=nspbps)
    try:
        fout[group].create_dataset('max_freq', (), data=max_freq)
        fout[group].create_dataset('uniform_width', (), data=uniform_width)
    except:
        pass

    # nv and nc bands are copied. Stochastic Pseudobands are constructed outside this range


def pseudoband(qshift, blocks_v, blocks_c, ifmax, params_from_parabands, fname_in=None, fname_out=None, fname_in_q=None,
               fname_out_q=None, fname_in_NNS=None, fname_out_NNS=None, nv=-1, nc=100, nspbps_v=2, nspbps_c=2,
               max_freq=0.0, uniform_width=None, single_band=False, copydirectly=True, verbosity=0, **kwargs):
    start = time.time()

    if qshift == 0:
        f_in = h5py.File(fname_in, 'r')
        f_out = h5py.File(fname_out, 'w')
        logger.info(f'fname_in = {fname_in}\n fname_out = {fname_out}\n ')
    elif qshift == 1:
        f_in = h5py.File(fname_in_q, 'r')
        f_out = h5py.File(fname_out_q, 'w')
        logger.info(f'fname_in = {fname_in_q}\n fname_out = {fname_out_q}\n ')
    elif qshift == 2:
        f_in = h5py.File(fname_in_NNS, 'r')
        f_out = h5py.File(fname_out_NNS, 'w')
        logger.info(f'fname_in = {fname_in_NNS}\n fname_out = {fname_out_NNS}\n ')

    nv = int(nv)
    nc = int(nc)

    mnband = f_in['mf_header/kpoints/mnband'][()]

    assert nv >= -1
    assert nc >= -1
    if nc > mnband - ifmax:
        nc = -1
        logger.warning('nc > mnband - ifmax, the original number of conduction states. Setting nc = -1')
    if nv > ifmax:
        nv = -1
        logger.warning('nv > ifmax, the original number of valence states. Setting nv = -1')

    if nv != -1:
        nslices_v = len(blocks_v)
        nspb_v = nslices_v * nspbps_v
        nb_out_v = nv + nspb_v
    else:
        nslices_v = 0
        nb_out_v = ifmax

    if nc != -1:
        nslices_c = len(blocks_c)
        nspb_c = nslices_c * nspbps_c
        nb_out_c = nc + nspb_c
    else:
        nslices_c = 0
        nb_out_c = mnband - ifmax

    if single_band:
        nspbps_v = 1
        nspbps_c = 1

    nk = f_in['mf_header/kpoints/nrk'][()]
    ngk = f_in['mf_header/kpoints/ngk'][()]
    cum_ngk = np.insert(np.cumsum(ngk), 0, 0)

    # Cannot read from this file if it exists due to different numbers of k-points in WFN and WFNq
    # Writing just for logging purposes
    if qshift == 0:
        phases_file = h5py.File('phases' + fname_out[0:-3] + '.h5', 'w')
    elif qshift == 1:
        phases_file = h5py.File('phases' + fname_out_q[0:-3] + '.h5', 'w')
    elif qshift == 2:
        phases_file = h5py.File('phases' + fname_out_NNS[0:-3] + '.h5', 'w')
    else:
        raise ValueError('bad value for qshift')

    f_out.copy(f_in['mf_header'], 'mf_header')
    f_out.create_group('wfns')
    f_out.copy(f_in['wfns/gvecs'], 'wfns/gvecs')
    f_out['mf_header/kpoints/mnband'][()] = nb_out_v + nb_out_c
    f_out['mf_header/kpoints/ifmax'][()] = nb_out_v

    f_out.create_group('pseudobands')
    f_out.create_group('pseudobands/conduction')
    f_out.create_group('pseudobands/valence')

    # fill in pseudobands params into these groups
    if nc == -1:
        try:
            nprot = params_from_parabands['nc'][()]
            nslice = params_from_parabands['n_subspaces'][()]
            nspbps = params_from_parabands['num_per_subspace'][()]
            fill_pseudoband_params(f_out, 'conduction', nprot, nslice, nspbps, max_freq, uniform_width)
        except:
            fill_pseudoband_params(f_out, 'conduction', nc, len(blocks_c), nspbps_c, max_freq, uniform_width)
    else:
        fill_pseudoband_params(f_out, 'conduction', nc, len(blocks_c), nspbps_c, max_freq, uniform_width)

    if nv != -1:
        fill_pseudoband_params(f_out, 'valence', nv, len(blocks_v), nspbps_v, max_freq, uniform_width)

    phases_file.copy(f_in['mf_header'], 'mf_header')
    phases_file.create_group('phases')

    def resize(file, name):
        file.move(name, name + '_orig_pb')
        shape = list(file[name + '_orig_pb'].shape)
        shape[-1] = nb_out_v + nb_out_c
        file.create_dataset(name, shape, dtype='d')
        del file[name + '_orig_pb']

    resize(f_out, 'mf_header/kpoints/occ')
    resize(f_out, 'mf_header/kpoints/el')

    if nv != -1 and nc != -1:
        logger.info('Copying {} protected bands'.format(nv + nc))
    elif nv == -1 and nc != -1:
        logger.info('Copying {} protected bands'.format(ifmax + nc))
    elif nv != -1 and nc == -1:
        logger.info('Copying {} protected bands'.format(mnband - ifmax + nv))

    shape = list(f_in['wfns/coeffs'].shape)
    shape[0] = nb_out_v + nb_out_c
    f_out.create_dataset('wfns/coeffs', shape, 'd')

    shape_phases = (ifmax - nv, nk, nspbps_v, 2,)
    phases_file.create_dataset('phases/coeffs', shape_phases, 'd')

    if copydirectly:  # always true, copying by chunks not yet implemented
        logger.warning('copydirectly=True. Copying all protected bands at once, can be slow if copying >1000 bands')
        if nv != -1 and nc != -1:
            f_out['wfns/coeffs'][nb_out_v - nv:nb_out_v + nc, :, :] = f_in['wfns/coeffs'][ifmax - nv:ifmax + nc, :, :]
            f_out['mf_header/kpoints/el'][:, :, nb_out_v - nv:nb_out_v + nc] =\
                f_in['mf_header/kpoints/el'][:, :, ifmax - nv:ifmax + nc]
        elif nv == -1 and nc != -1:
            f_out['wfns/coeffs'][0:nb_out_v + nc, :, :] = f_in['wfns/coeffs'][0:ifmax + nc, :, :]
            f_out['mf_header/kpoints/el'][:, :, 0:nb_out_v + nc] = f_in['mf_header/kpoints/el'][:, :, 0:ifmax + nc]
        elif nv != -1 and nc == -1:
            f_out['wfns/coeffs'][nb_out_v - nv:None, :, :] = f_in['wfns/coeffs'][ifmax - nv:None, :, :]
            f_out['mf_header/kpoints/el'][:, :, nb_out_v - nv:None] =\
                f_in['mf_header/kpoints/el'][:, :, ifmax - nv:None]

    if nv != -1:
        logger.info('Creating {} valence pseudobands'.format(nspb_v))
        ib = nb_out_v - nv - 1

        for b in blocks_v:

            num_bands_in = b[0] - b[1] + 1

            if verbosity > 1:
                logger.info(f'slice_index, slice: {ib - (nb_out_v - nv - 1), b}')

            if single_band:
                band_avg = b[0] + (b[1] - b[0]) // 2
                f_out['wfns/coeffs'][ib, :, :] = (f_in['wfns/coeffs'][band_avg, :, :]
                                                  * np.sqrt(float(b[1] - b[0] + 1)))
                f_out['mf_header/kpoints/el'][:, :, ib] = f_in['mf_header/kpoints/el'][:, :, band_avg].mean(axis=-1)
                ib -= 1

            elif nspbps_v == 1:
                if b[0] == 1:
                    f_out['wfns/coeffs'][ib, :, :] = f_in['wfns/coeffs'][0, :, :].sum(axis=0)
                    f_out['mf_header/kpoints/el'][:, :, ib] = f_in['mf_header/kpoints/el'][:, :, 0].mean(axis=-1)
                elif b[0] == 0:
                    continue
                else:
                    f_out['wfns/coeffs'][ib, :, :] = f_in['wfns/coeffs'][b[1]:b[0] + 1, :, :].sum(axis=0)
                    f_out['mf_header/kpoints/el'][:, :, ib] = f_in['mf_header/kpoints/el'][:, :, b[1]:b[0] + 1].mean(
                        axis=-1)

                # normalization: <SPB|SPB> = num_band_in * nk
                logger.info(
                    f'''num_bands_in * nk, norm(SPB)**2: {num_bands_in * nk}, 
                {np.linalg.norm(f_out['wfns/coeffs'][ib, :, :].view(np.complex128)) ** 2}''')
                assert abs(
                    np.linalg.norm(f_out['wfns/coeffs'][ib, :, :].view(np.complex128)) ** 2 - num_bands_in * nk) <= 1e-9

                ib -= 1

            else:  # N_ξ > 1
                if b[0] == 0:  # reached the end
                    continue

                else:
                    coeffs = f_in['wfns/coeffs'][b[1]:b[0] + 1, :, :, :].view(np.complex128)
                    el = f_in['mf_header/kpoints/el'][:, :, b[1]:b[0] + 1].mean(axis=-1)

                    for ispb in range(nspbps_v):
                        # Phases are normalized to the DOS / sqrt(number_pseudobands)
                        phases = np.random.random((num_bands_in, nk,))
                        phases = np.exp(2 * np.pi * 1.0j * phases) / np.sqrt(float(nspbps_v))

                        phases_file['phases/coeffs'][b[1]:b[0] + 1, :, ispb, :] = phases.view(np.float64).reshape(
                            (phases.shape + (2,)))
                        # NOTE: nk in WFN and WFNq are often different, cannot reuse phases from file

                        if verbosity > 1:
                            logger.info(f'phases.shape for SPB {ib}: {phases.shape}')
                            logger.info(f'coeffs.shape for SPB {ib}: {coeffs.shape}')

                        # Make sure we do a complex mult., and then view result as float
                        spb = [np.tensordot(coeffs[:, :, cum_ngk[k]:cum_ngk[k + 1]], phases[:, k], axes=(0, 0)) for k in
                               range(nk)]
                        spb = np.concatenate(spb, axis=-2)

                        f_out['wfns/coeffs'][ib, :, :] = spb.view(np.float64)
                        f_out['mf_header/kpoints/el'][:, :, ib] = el

                        # normalization: <SPB|SPB> = num_band_in * nk / num_per_slice
                        logger.info(
                            f'''num_bands_in*nk/nspbps_v, norm(SPB)**2: {num_bands_in / float(nspbps_v) * nk}, 
{np.linalg.norm(spb) ** 2}''')
                        assert abs(np.linalg.norm(spb) ** 2 - num_bands_in * nk / float(nspbps_v)) <= 1e-9

                        ib -= 1

    if qshift:
        f_out.close()
        phases_file.close()

        f_in.close()

        end = time.time()
        logger.info(f'Done! Time taken: {round(end - start, 2)} sec\n\n\n')
        return

    if nc != -1:
        logger.info('Creating {} conduction pseudobands'.format(nspb_c))
        ib = nb_out_v + nc

        for b in blocks_c:

            num_bands_in = b[1] - b[0] + 1

            if verbosity > 1:
                logger.info(f'slice_index, slice: {ib - (nb_out_v + nc), b}')

            if single_band:
                band_avg = b[0] + (b[1] - b[0]) // 2
                f_out['wfns/coeffs'][ib, :, :] = (f_in['wfns/coeffs'][band_avg, :, :]
                                                  * np.sqrt(float(b[1] - b[0] + 1)))
                f_out['mf_header/kpoints/el'][:, :, ib] = f_in['mf_header/kpoints/el'][:, :, band_avg].mean(axis=-1)
                ib += 1

            elif nspbps_c == 1:
                f_out['wfns/coeffs'][ib, :, :] = f_in['wfns/coeffs'][b[0]:b[1] + 1, :, :].sum(axis=0)
                f_out['mf_header/kpoints/el'][:, :, ib] = f_in['mf_header/kpoints/el'][:, :, b[0]:b[1] + 1].mean(
                    axis=-1)
                if verbosity > 1:
                    avgen = f_in['mf_header/kpoints/el'][:, :, b[0]:b[1] + 1].mean(axis=-1)
                    logger.info(f'energy of slice {b}: {avgen}')

                # normalization: <SPB|SPB> = num_band_in * nk
                logger.info(
                    f'''num_bands_in, norm(SPB)**2: {num_bands_in * nk}, 
{np.linalg.norm(f_out['wfns/coeffs'][ib, :, :].view(np.complex128)) ** 2}''')
                assert abs(
                    np.linalg.norm(f_out['wfns/coeffs'][ib, :, :].view(np.complex128)) ** 2 - num_bands_in * nk) <= 1e-9

                ib += 1

            else:
                coeffs = f_in['wfns/coeffs'][b[0]:b[1] + 1, :, :, :].view(np.complex128)
                el = f_in['mf_header/kpoints/el'][:, :, b[0]:b[1] + 1].mean(axis=-1)
                num_bands_in = b[1] - b[0] + 1
                for ispb in range(nspbps_c):
                    # no phases file for conduction states (would be far too large)
                    # Phases are normalized to the DOS / sqrt(number_pseudobands)
                    phases = np.random.random((num_bands_in, nk,))
                    phases = np.exp(2 * np.pi * 1.0j * phases) / np.sqrt(float(nspbps_c))

                    if verbosity > 1:
                        logger.info(f'phases.shape for SPB {ib}: {phases.shape}')
                        logger.info(f'coeffs.shape for SPB {ib}: {coeffs.shape}')

                    # Make sure we do a complex mult., and then view result as float
                    spb = np.concatenate(
                        [np.tensordot(coeffs[:, :, cum_ngk[k]:cum_ngk[k + 1]], phases[:, k], axes=(0, 0)) for k in
                         range(nk)], axis=-2)

                    f_out['wfns/coeffs'][ib, :, :] = spb.view(np.float64)
                    f_out['mf_header/kpoints/el'][:, :, ib] = el

                    # normalization: <SPB|SPB> = num_band_in / num_per_slice * nk
                    logger.info(
                        f'''num_bands_in*nk/nspbps_c, norm(SPB)**2: {num_bands_in / float(nspbps_c) * nk}, 
{np.linalg.norm(spb) ** 2}''')
                    assert abs(np.linalg.norm(spb) ** 2 - num_bands_in * nk / float(nspbps_c)) <= 1e-9

                    ib += 1

    f_out.close()
    phases_file.close()

    f_in.close()

    end = time.time()
    logger.info(f'Done! Time taken: {round(end - start, 2)} sec\n\n\n')


if __name__ == "__main__":
    import argparse

    desc = '''Constructs stochastic pseudobands given an input WFN. Bands with
    energies up to a protection window E0 are copied without modification, and
    states higher in energy are aggregated into subspaces and represented with
    pseudobands, each one spanning the energy range [En, En * E_frac], where En
    is the energy of the first state in the subspace, and E_frac is a constant.'''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--fname_in', help='Input WFN.h5, in HDF5 format', required=True)
    parser.add_argument('--fname_in_q', help='Input WFNq.h5, in HDF5 format', required=True)
    parser.add_argument('--fname_in_NNS', help='Output WFN.h5 with pseudobands, in HDF5 format')
    parser.add_argument('--fname_out', help='Output WFN.h5 with pseudobands, in HDF5 format', required=True)
    parser.add_argument('--fname_out_q', help='Output WFNq.h5 with pseudobands, in HDF5 format', required=True)
    parser.add_argument('--fname_out_NNS', help='Output WFNq.h5 with pseudobands, in HDF5 format')
    parser.add_argument('--NNS', default=0, help='Using separate NNS WFNq?')
    parser.add_argument('--nv', '--N_P_val', type=int, default=-1,
                        help='Number of protected valence bands counting from VBM.')
    parser.add_argument('--nc', '--N_P_cond', type=int, default=100,
                        help='Number of protected conduction bands counting from CBM.')
    parser.add_argument('--nslice_v', '--N_S_val', type=int, default=10,
                        help='Number of subspaces spanning the total energy range of the valence bands.')
    parser.add_argument('--uniform_width', type=float, default=None,
                        help=(
                            'Constant width accumulation window (Ry) for conduction slices with energies <= max_freq.'))
    parser.add_argument('--nslice_c', '--N_S_cond', type=int, default=100,
                        help='Number of subspaces spanning the total energy range of the conduction bands.')
    parser.add_argument('--max_freq', type=float, default=0.0,
                        help=(
                            '''Maximum energy (Ry) before coarse slicing kicks in for conduction SPBs. 
                            This should be at least the maximum frequency for which you plan to evaluate epsilon.'''))
    parser.add_argument('--nspbps_v', '--N_xi_val', type=int, default=2,
                        help='Number of stochastic pseudobands per valence slice. Must be at least 2.')
    parser.add_argument('--nspbps_c', '--N_xi_cond', type=int, default=2,
                        help='Number of stochastic pseudobands per conduction slice. Must be at least 2.')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Set verbosity level')
    parser.add_argument('--single_band', default=False, action='store_true',
                        help='Use a single band instead of a stochastic combination')

    args = parser.parse_args()

    logging.basicConfig(filename=vars(args)['fname_out'][0:-3] + '.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)

    logger.info('''\n
    ____________________________________________________________________________________________________________\n
    ____________________________________________________________________________________________________________\n
    ____________________________________________________________________________________________________________\n\n
                                              STOCHASTIC PSEUDOBANDS\n\n
    ____________________________________________________________________________________________________________\n
    ____________________________________________________________________________________________________________\n
    ____________________________________________________________________________________________________________\n''')

    logger.info(vars(args))

    out = check_and_block(**vars(args))

    pseudoband(0, *out, **vars(args))

    pseudoband(1, *out, **vars(args))

    if int(vars(args)['NNS']) == 1:
        pseudoband(2, *out, **vars(args))

