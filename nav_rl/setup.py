from setuptools import find_packages, setup

package_name = 'nav_rl'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali_Karimzade',
    maintainer_email='alikarimzade29@gmail.com',
    description='Navigation + TD3 + HER',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_node = nav_rl.rl_node:main'
        ],
    },
)
