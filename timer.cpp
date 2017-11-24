#include <vector>
#include <chrono>
using namespace std::chrono;
using std::vector;

class Timer {
	private:
	double started, delayTime;
	public:
	vector <double> times;
	Time (const double _delayTime) delayTime : _delayTime {}
	void start () { started = system_clock::now (); }
	bool up () = { return  (system_clock::now () - started).count () >= delayTime; }
	double current () = { return  (system_clock::now () - started).count (); }
	void reset () { times.push_back (this->current());}
};

