#include <vector>
#include <chrono>

using namespace std::chrono;
using std::vector;

class Timer {
	time_point <system_clock> started;
	double delayTime;
//	vector <double> times;
public:
	Timer () {}
	Timer (const double _delayTime) :delayTime (_delayTime) {}
	void start (const double _delayTime) {
		delayTime = _delayTime;
		started = system_clock::now ();
	}
	bool up () {
		return current() >= delayTime;
	}
	double current () {
		return  duration<double>(system_clock::now () - started).count ();
	}
	void reset () {
//		times.push_back (this->current());
		started = system_clock::now ();
	}
};

