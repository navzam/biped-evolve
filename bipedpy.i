%include "carrays.i"
%array_class(float,floatArray);

%newobject *::copy;
%module bipedpy
%{
#include "biped_py.h"
%}

/*
class feature_detector {
public:
static float end_goal(mazenav* mn);
static float start_dist(mazenav* mn);
static float closest_goal(mazenav* mn);
static float endx(mazenav* mn);
static float endy(mazenav* mn);
static float midx(mazenav* mn);
static float midy(mazenav* mn);
static float spd(mazenav* mn);
static float coll(mazenav* mn);
static float turn(mazenav* mn);
static float state_entropy(mazenav* mn);
};
*/

class bipedsim {
public: 
 void make_random();

 static void seed(int sd);
 static void random_seed();
 bipedsim();

  bipedsim* copy();
  int complexity();

  void map();
  static void initmaze(const char* mazefile);
  void mutate();
  bool isvalid();
  void clear();
  double distance(bipedsim* other);
  void init_rand();
 
  void save(const char *fname);
  void load_new(const char *fname);

  float get_x(); 
  float get_y();
  bool solution();
  ~bipedsim();

};
