all:
	python setup.py build_clib
	python setup.py build_ext --inplace
clean:
	rm -rf build/
	rm -f pysmoothspl/*.c
