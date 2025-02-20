This project employs _local_ Large Language Models to summarise a given list of .pdf documents. The codebase is largely based on [this](https://github.com/debugverse/debugverse-youtube/tree/main/summarize_huge_documents_kmeans) work.

It achieves this in four fundamental steps:

1- it loads the .pdf document and splits it in chunks of a given size

2- it embeds the chunks in a latent space (https://huggingface.co/BAAI/bge-large-en-v1.5)

3- it performs k-means clustering in the latent space, to clump chunks by similarity

4- it summarizes the resulting clustered chunks into one final text

### Installation & requirements
Software requirements:
- python (miniconda recommended)
- pip
- Ollama (for your local LLM - alternatively an API key for your favourite pay-per-use model)

Begin by installing Ollama, setting it up and testing it in your command line. It is useful to pull the model you are going to use for the exercise. 
```
% ollama pull llama3.2:latest
```

```
% ollama run llama3.2:latest

>>> how are you today?

I'm just a language model, so I don't have feelings or emotions like humans do. However, I'm functioning properly and ready to assist you 

with any questions or tasks you may have! How can I help you today?
```

Next you can clone this github repository, and set up your conda environment using the included requirements.txt file

```
conda create --name llm_summarizer --file requirements.txt
```

You should now be ready to go!

### Usage

Drop your papers in the "papers" folder, then run the script. This will generate one summary for each file contained in the papers folder and write those to identically named .txt files in the main folder.

Note: depending on your computer's hardware, this might be very slow or fail completely. All LLM models supported by Ollama have documentation on the [website](https://ollama.com), make sure that your computer has enough RAM and/or VRAM so that the model fits in your memory. Models larger than your (V)RAM will not work.

### Hyperparameters

This project has three fundamental parameters:
- the chosen LLM model (default: model="llama3.2")
- the size of the document chunks (default: chunk_size=8000)
- the number of clusters (default: num_clusters=6)

The values have been fine tuned manually to reproduce reasonable results. Multiple LLM models were tested, none larger than 8GB in size.

### References

Instructional videos on LLMs and summarizing:

https://www.youtube.com/watch?v=doRpfmXncEE - simplest approach, feeds documents directly to the LLMs' context and prompts

https://www.youtube.com/watch?v=_XayFqTk3EY - project that allows taking to your pdf, so you can ask questions about the paper

https://www.youtube.com/watch?v=Gn64NNr3bqU - explainer of the clustering method we use here

https://www.youtube.com/watch?v=qaPMdcCqtWk - multiple methods outlined one by one


Further reading:

https://aclanthology.org/2020.acl-main.463.pdf - what do LLMs even learn, precisely?
https://llm-calc.rayfernando.ai - a simple tool that helps you figure out how much memory you need for your LLM
https://mlco2.github.io/impact/ - another tool that allows you to estimate the CO2 emissions associated with your AI

