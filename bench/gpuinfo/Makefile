CXXFLAGS = -O2 -fobjc-arc -x objective-c++ -std=c++17
LDLIBS   = -framework Foundation -framework Metal

gpuinfo: gpuinfo.m
	clang++ $(CXXFLAGS) $(LDLIBS) -o gpuinfo gpuinfo.m

clean:
	rm -f gpuinfo

.PHONY: clean
