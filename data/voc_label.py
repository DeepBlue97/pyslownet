#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen abs path of images
import os
from os import listdir, getcwd
from os.path import join, abspath

iterVersions = ['0.1.0']
imageSetTypes = ['val','test','train',]
cwdir = getcwd()
print('cwd = '+cwdir)
#imageSetPath = join(cwdir, '..', '/0_train/VocLabel/Ver#{version}/ImageSets/Main')
imageSetPath = cwdir + '/VocLabel/Ver#{version}/ImageSets/Main'
#imageOutPath = join(cwdir, '..', '/0_train/VocLabel/Ver#{version}/JPEGImages')
imageOutPath = cwdir + '/VocLabel/Ver#{version}/JPEGImages'

for image_set in imageSetTypes:
    image_set_file = image_set + '.txt'
    list_file = open(image_set_file, 'w')
    for version in iterVersions:
        print(imageOutPath)
        ids_file = join(imageSetPath, image_set_file).replace('#{version}', version)
        image_ids = open(ids_file).read().strip().split()
        for image_id in image_ids:
            image_out_path = join(imageOutPath, image_id).replace('#{version}', version)
            list_file.write('%s.jpg\n'%(abspath(image_out_path)))
        # end read ids_file
    # end read list_file
    list_file.close()
#end deal all image sets
