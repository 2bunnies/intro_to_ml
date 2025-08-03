###############################################

# Makefile

###############################################

.PHONY: venv

venv/bin/activate: requirements.txt
	test -d venv || virtualenv -p python venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

venv: venv/bin/activate

clean:
	rm -f *.pyc
	rm -rf venv