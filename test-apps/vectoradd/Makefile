all:
	nvcc -ccbin=$(CXX) -O0 -g -G -D_FORCE_INLINES -arch=sm_35 vectoradd.cu -o vectoradd::

clean:
	rm -f vectoradd
