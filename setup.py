from setuptools import setup, find_packages

setup(
    name="Tabular_to_Neo4j",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.0.10",
        "openai>=1.0.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "langdetect>=1.0.9",
        "chardet>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0",
            "tox>=3.24.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
    },
    package_data={
        "Tabular_to_Neo4j": ["prompts/*.txt"],
    },
    entry_points={
        "console_scripts": [
            "tabular2neo4j=Tabular_to_Neo4j.main:main",
        ],
    },
    python_requires=">=3.9",
    author="Fabio Yáñez Romero",
    author_email="fabioyanezromero@gmail.com",
    description="A tool to analyze CSV files and infer Neo4j graph database schemas",
    long_description=open("Tabular_to_Neo4j/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FabioYanezRomero/tabular_neo4j",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
