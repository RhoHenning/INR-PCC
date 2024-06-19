import os
import subprocess
import numpy as np

metrics = {
    'mseF(p2point):'     : 'D1',
    'mseF,PSNR(p2point):': 'D1 PSNR',
    'mseF(p2plane):'     : 'D2',
    'mseF,PSNR(p2plane):': 'D2 PSNR',
    'c[0],F': 'Y',
    'c[1],F': 'U',
    'c[2],F': 'V',
    'c[0],PSNRF:': 'Y PSNR',
    'c[1],PSNRF:': 'U PSNR',
    'c[2],PSNRF:': 'V PSNR'
}
pc_error_path = os.path.join(os.path.dirname(__file__), 'pc_error')

def mean_psnr(psnr_1, psnr_2, psnr_3):
    return -10 * np.log10((np.power(10, -psnr_1 / 10) + np.power(10, -psnr_2 / 10) + np.power(10, -psnr_3 / 10)) / 3)

def distortion(path_1, path_2, resolution, has_colors=True):
    config = (
        f' --fileA={path_1}' +
        f' --fileB={path_2}' +
        f' --color={int(has_colors)}' +
        f' --resolution={resolution}' +
        ' --dropdups=2' +
        ' --neighborsProc=1'
    )
    command = pc_error_path + config

    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    results = {}
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        words = ' '.join(line.split()).split()
        if len(words) >= 3 and words[0] + words[1] in metrics:
            results[metrics[words[0] + words[1]]] = float(words[-1])
        c = subp.stdout.readline()
    if has_colors:
        results['YUV'] = (results['Y'] + results['U'] + results['V']) / 3
        results['YUV PSNR'] = mean_psnr(results['Y PSNR'], results['U PSNR'], results['V PSNR'])
    return results
