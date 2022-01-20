
doc : docs/index.html

docs/index.html : logiclocking/* docs/templates/*
	pdoc --html logiclocking --force --template-dir docs/templates
	cp html/logiclocking/* docs

test :
	python3 -m unittest

test_% :
	python3 -m unittest logiclocking/tests/test_$*.py

coverage :
	coverage run -m unittest
	coverage html

dist : setup.py
	rm -rf dist/* build/* circuitgraph.egg-info
	python3 setup.py sdist bdist_wheel

test_upload: dist
	python3 -m twine upload --repository testpypi dist/*

upload : dist
	python3 -m twine upload dist/*

install:
	pip3 install .

install_editable :
	pip3 install -e .
