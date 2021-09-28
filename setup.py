import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BB_gym_Envs", # Replace with your own username
    version="0.0.1",
    author="Dawson Horvath",
    author_email="horvath.dawson@gmail.com",
    description="ignition-gym environments for baesian balacning development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Baesian-Balancer/BB_gym_Envs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'SIMP@git+https://github.com/Baesian-Balancer/SIMP.git@v1.1',
          'gym-ignition',
    ],
    python_requires='>=3.8',
)
