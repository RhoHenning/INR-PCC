import os
import subprocess

metrics = {
    'positions bitstream size': 'geometry',
    'colors bitstream size'   : 'attribute',
    'Total bitstream size'    : 'total'
}
tmc3_path = os.path.join(os.path.dirname(__file__), 'tmc3')

def octree_raht_config(pqs, qp):
    return (
        ' --mode=0' +
        ' --trisoupNodeSizeLog2=0' +
        ' --mergeDuplicatedPoints=1' +
        ' --neighbourAvailBoundaryLog2=8' +
        ' --intra_pred_max_node_size_log2=6' +
        f' --positionQuantizationScale={pqs}' +
        ' --maxNumQtBtBeforeOt=4' +
        ' --minQtbtSizeLog2=0' +
        ' --planarEnabled=1' +
        ' --rahtPredictionSearchRange=50000' +
        ' --planarEnabled=0' +
        ' --planarModeIdcmUse=0' +
        ' --convertPlyColourspace=1' +
        ' --transformType=0' +
        f' --qp={qp}' +
        ' --qpChromaOffset=-2' +
        ' --bitdepth=8' +
        ' --attrOffset=0' +
        ' --attrScale=1' +
        ' --attribute=color'
    )

def encode(cloud_path, bin_path, pqs=None, qp=None, encode_colors=True):
    config = (
        octree_raht_config(pqs, qp) +
        f' --uncompressedDataPath={cloud_path}' +
        f' --compressedStreamPath={bin_path}' +
        f' --disableAttributeCoding={int(not encode_colors)}'
    )
    command = tmc3_path + config
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    results = {}
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        words = ' '.join(line.split()).split()
        if len(words) >= 4 and ' '.join(words[:3]) in metrics:
            results[metrics[' '.join(words[:3])]] = eval(words[3])
        c = subp.stdout.readline()
    return results

def decode(bin_path, cloud_path):
    config = (
        ' --mode=1' +
        ' --convertPlyColourspace=1' +
        f' --compressedStreamPath={bin_path}' +
        f' --reconstructedDataPath={cloud_path}' +
        ' --outputBinaryPly=0'
    )
    command = tmc3_path + config
    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    results = {}
    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')
        words = ' '.join(line.split()).split()
        if len(words) >= 4 and ' '.join(words[:3]) in metrics:
            results[metrics[' '.join(words[:3])]] = eval(words[3])
        c = subp.stdout.readline()
    return results