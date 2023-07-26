import os

import launch

if not launch.is_installed('send2trash'):
    print('running pip install send2trash')
    launch.run_pip('install Send2Trash', 'requirement for images-browser')

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'requirements.txt')

with open(req_file) as f:
    for lib in f:
        lib = lib.strip()
        if not launch.is_installed(lib):
            print('pip install {}'.format(lib))
            launch.run_pip(f'install {lib}', f'sd-webui requirement: {lib}')
