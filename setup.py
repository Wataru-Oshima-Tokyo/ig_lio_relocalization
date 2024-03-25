import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'ig_lio_relocalization'
library = 'ig_lio_relocalization/library'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, library],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        ('share/' + package_name + '/params', glob('params/*')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='wataru.bb.tokyo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'ig_lio_relocalization_node = ig_lio_relocalization.ig_lio_relocalize:main',
             'ig_lio_tf_fusion_node = ig_lio_relocalization.ig_lio_transform_fusion:main',
             'visualization_test_node = ig_lio_relocalization.visualization_test:main',
        ],
    },
)
