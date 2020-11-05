# An Introduction to Machine Learning Fundamentals

To run this project, either use a locally installed jupyter with tensorflow 2, keras, sklearn, numpy and pygraphviz, or simply run `docker-compose up` in this directory and use the provided docker image. It will have everything you need.

If you're new to Jupyter notebooks, running this command will create a running notebook server on your machine, exposed at port 8888.
After running this command, you can go to http://127.0.0.1:8888 to see your server.
You'll notice it will require a token for you to access it - you can find that token in the output logs of the `docker-compose` command. It will look something like this:

	http://127.0.0.1:8888/?token=ecf605284aee8faf1e7edd6b647a7ba5d2bc3b539938e07e

You can in some terminals also just click the link to open it and automatically log in.

## Exercises

The exercises are labelled with numbers, followed by EX, and I recommend starting with the first.

## Guides

There's also a bunch of guides attempting to explain some of the more fundamental concepts in a little more detail, should you struggle in some places.

## Tips

If you're new to how jupyter notebooks work, I recommend the interactive tutorial at: https://jupyter.org/try (pick the classic one). It is a bit lengthy, and covers most features. We'll only need a subset, so you can also just experiment here.

Have fun!

## Dependencies for manual install

If you're looking to install this manually on your machine you'll need:

* Python 3.6, 3.7 or 3.8
* Jupyter
* Graphviz installed (For windows, remember to add it to your PATH environment variable)
* pip3

### Graphviz notes

For windows, go here: https://graphviz.gitlab.io/_pages/Download/Download_windows.html
For mac: You can install graphviz using homebrew
For ubuntu: Just apt-get install graphviz

### Python dependencies

* pygraphviz
* numpy
* tensorflow==2.3.1
* sklearn
* matplotlib

On windows, you might be required to install the visual studio c++ build tools in order to install and build pygraphviz.

## Running the notebook

If you've installed this manually, you can run it either through your preferred terminal or Anaconda shell (if you've installed conda). Navigate to this repository on your machine and execute `jupyter notebook` to open a new notebook in that working directory, and you should be set.
