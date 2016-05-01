#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include "noveltyset.h"
#include "neat.h"
#include "network.h"
#include "population.h"
#include "organism.h"
#include "genome.h"
#include "species.h"
#include "datarec.h"
#include "maze.h"
#include "population_state.h"

using namespace std;
using namespace NEAT;

void test_ensemble(const char* classfile);

void enumerate_behaviors(const char* name,long long parm,const char* outname,int count);
void mutate_genome(Genome* new_genome,bool traits=false);

float contained_dist(float x,float y);
bool contained(float x,float y);

void set_age_objective(bool ao); 
void set_evaluate(bool val);
void set_extinction(bool _ext);
void set_random_replace(bool val);
void  set_constraint_switch(bool val);
void set_aoi(bool val);
void set_nov_measure(string m);
void set_fit_measure(string m);

bool set_no_collision(bool no); 
bool set_reach_onepoint(bool ro);
void set_mcmaze(string s);
void set_minimal_criteria(bool mc);
void set_samples(int s);
void set_timesteps(int s);
void set_seed(string s);
void set_goal_attract(bool s);
//generational maze experiments
typedef int (*successfunc)(population_state* ps);
typedef int (*epochfunc)(population_state* ps,int generation,successfunc sf);

Population *classifier_generational(char* output_dir,const char* classfile,int param, const char *genes, int gens, bool novelty); 
int classifier_success_processing(population_state* pstate);
int classifier_generational_epoch(population_state* pstate,int generation);

int generalized_generational_epoch(population_state* pstate,int generation,successfunc success_processing);
void destroy_organism(Organism* curorg);

//int maze_generational_epoch(Population **pop,int generation,data_rec& Record,noveltyarchive& archive,bool novelty);

//Walker novelty steady-state 
noveltyitem* classifier_novelty_map(Organism *org,data_record* record=NULL);

#endif
