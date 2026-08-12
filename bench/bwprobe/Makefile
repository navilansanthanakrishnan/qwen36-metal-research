# bwprobe uses a C++ raw string literal for the MSL source, so it compiles as
# Objective-C++ despite the .m extension.
CXXFLAGS = -O2 -fobjc-arc -x objective-c++ -std=c++17
LDLIBS   = -framework Foundation -framework Metal

bwprobe: bwprobe.m
	clang++ $(CXXFLAGS) $(LDLIBS) -o bwprobe bwprobe.m

clean:
	rm -f bwprobe

.PHONY: clean
