#include "read_data.h"
#include "experiments.h"
#include "noveltyset.h"

#include "datarec.h"
#include "maze.h"
#include "graph.h"

#include "histogram.h"
#include "calc_evol.h"
#include "genome.h"
//#define DEBUG_OUTPUT 1

#include <algorithm>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <math.h>

#include "population.h"
#include "population_state.h"
#include "alps.h"
//#include "modularity/modularity.hpp"

static Environment* env;
static vector<vector<float> > classifier_train_data;
static vector<vector<float> > classifier_valid_data;
static vector<vector<float> > classifier_test_data;
static vector<Environment*> envList;
static vector<Environment*> mcList;
static ofstream *logfile=NULL;

static vector<float> best_fits;
plot front_plot;
plot fitness_plot;
plot behavior_plot;
//void modularity(Organism* org,char* fn);

/*double modularity_score(Genome* start_genome) {
Network* test_net=start_genome->genesis(0);
Graph* g=test_net->to_graph();    
std::vector<std::set<Graph::vertex_descriptor> > mods;
float modularity = mod::h_modules(*g,mods);
//cout << "modularity: " << modularity << endl;
return modularity;
delete test_net;
delete g;
}*/
using namespace std;
enum novelty_measure_type { novelty_fitness, novelty_sample, novelty_accum, novelty_sample_free };
static novelty_measure_type novelty_measure = novelty_sample;

enum fitness_measure_type { fitness_uniform,fitness_goal, fitness_drift, fitness_std,fitness_rnd,fitness_spin,fitness_changegoal,fitness_collisions,fitness_reachone ,fitness_aoi,fitness_collgoal};
static fitness_measure_type fitness_measure = fitness_goal;
static bool mc_no_collision=false;
static bool mc_reach_onepoint=false;
bool age_objective=false;
bool population_dirty=false;

static bool regression=true;

static bool extinction=true;
bool set_no_collision(bool no) { mc_no_collision=no; }
bool set_reach_onepoint(bool ro) { mc_reach_onepoint=ro; }
bool get_age_objective() { return age_objective; }
void set_age_objective(bool ao) { age_objective=ao; }
void set_extinction(bool _ext) {
  extinction=_ext;
}
static Point extinction_point(0.0,0.0);
static float extinction_radius=50.0;
static int change_extinction_length=5;

void change_extinction_point() {
  float minx,maxx,miny,maxy;
  envList[0]->get_range(minx,miny,maxx,maxy);

  extinction_point.x = randfloat()*(maxx-minx)+minx;
  extinction_point.y = randfloat()*(maxy-miny)+miny;
  if (extinction)
    population_dirty=true;
}
bool extinct(Point k) {
  if (k.distance(extinction_point)<extinction_radius)
    return true;
  return false;
}


static Point changing_goal(0.0,0.0);
static int change_goal_length=5;


void change_goal_location() {
  static int count=0;
  static float vx=0.0;
  static float vy=0.0;
  static float newx;
  static float newy;
  float minx,maxx,miny,maxy;
  envList[0]->get_range(minx,miny,maxx,maxy);

  if (count==0) {
    newx = randfloat()*(maxx-minx)+minx;
    newy = randfloat()*(maxy-miny)+miny;
    changing_goal.x=newx;
    changing_goal.y=newy;
  }

  if (count%10==0) {
    newx = randfloat()*(maxx-minx)+minx;
    newy = randfloat()*(maxy-miny)+miny;
    vx=(newx-changing_goal.x)/10.0;
    vy=(newy-changing_goal.y)/10.0;
  }

  count++;

  changing_goal.x+=vx;
  changing_goal.y+=vy;
  if (changing_goal.x<minx) changing_goal.x=minx;
  else if (changing_goal.x>maxx) changing_goal.x=maxx;
  if (changing_goal.y<miny) changing_goal.y=miny;
  else if (changing_goal.y>maxy) changing_goal.y=maxy;


  if (fitness_measure==fitness_changegoal)
  {
    cout << "New goal: " << changing_goal.x << " " << changing_goal.y << endl;
    population_dirty=true;
  }

}




static int number_of_samples = 1;
static int simulated_timesteps = 400;
bool seed_mode = false;
char seed_name[100]="";
bool minimal_criteria=false;
bool evaluate_switch=false;
static bool goal_attract=true;

static bool activity_stats=false;

static bool constraint_switch=false;
static bool area_of_interest=false;
static bool rand_repl=false;
void set_evaluate(bool val) {
  evaluate_switch=val;
}
void set_random_replace(bool val)
{
  rand_repl = val;
}

void set_aoi(bool val)
{
  area_of_interest=val;
}

void  set_constraint_switch(bool val)
{
  constraint_switch=val;
}
void set_minimal_criteria(bool mc)
{
  minimal_criteria=mc;
}

void set_goal_attract(bool ga)
{
  goal_attract=ga;
}

void set_samples(int s)
{
  number_of_samples=s;
}

void set_timesteps(int s)
{
  simulated_timesteps=s;
}

void set_seed(string s)
{
  strcpy(seed_name,s.c_str());
  if (strlen(seed_name)>0)
    seed_mode=true;
}

void set_fit_measure(string m)
{
  if (m=="uniform")
    fitness_measure=fitness_uniform;
  if (m=="reachone")
    fitness_measure=fitness_reachone;
  if (m=="rnd")
    fitness_measure=fitness_rnd;
  if (m=="std")
    fitness_measure=fitness_std;
  if (m=="drift")
    fitness_measure=fitness_drift;
  if (m=="goal")
    fitness_measure=fitness_goal;
  if (m=="spin")
    fitness_measure=fitness_spin;
  if (m=="changegoal")
    fitness_measure=fitness_changegoal;
  if (m=="collisions")
    fitness_measure=fitness_collisions;
  if (m=="aoi")
    fitness_measure=fitness_aoi;
  if (m=="nocollide_goal")
    fitness_measure=fitness_collgoal;
  cout << "Fitness measure " << fitness_measure << endl;
}

void set_nov_measure(string m)
{
  if (m=="fitness")
    novelty_measure=novelty_fitness;
  if (m=="std" || m=="sample")
    novelty_measure=novelty_sample;
  if (m=="accum")
    novelty_measure=novelty_accum;
  if (m=="sample_free")
    novelty_measure=novelty_sample_free;
  cout << "Novelty measure " << novelty_measure << endl;
}

char output_dir[200]="";


static int param=-1;
static int push_back_size = 20;

//used for discretization studies
double discretize(double x,long bins,double low, double high)
{
  double norm = x-low;
  double binsize = (high-low)/bins;
  int bin = (int)(norm/binsize);
  if (bin==bins)
    bin--;
  double result = (double)binsize*bin+binsize/2.0+low;
  return result;
}

long powerof2(int num)
{
  long x=1;
  if (num==0) return 1;
  for (int y=0; y<num; y++)
    x*=2;
  return x;
}

//novelty metric for maze simulation
float maze_novelty_metric(noveltyitem* x,noveltyitem* y)
{
  float diff = 0.0;
  for (int k=0; k<(int)x->data.size(); k++)
  {
    diff+=hist_diff(x->data[k],y->data[k]);
  }
  return diff;
}

static void read_in_environments(const char* mazefile, vector<Environment*>& envLst)
{
  ifstream listfile(mazefile);

  while (!listfile.eof())
  {
    string filename;
    getline(listfile,filename);
    if (filename.length() == 0)
      break;
    cout << "Reading maze: " << filename << endl;
    Environment* new_env = new Environment(filename.c_str());
    envLst.push_back(new_env);
  }

}

#define MAX_NICHES 300000
class passive_niche {
public:
        
  //niches, whether niches have been explored
  vector<Organism*> niches[MAX_NICHES];
  bool explored[MAX_NICHES];
  int order[MAX_NICHES];
  //params
  int niche_size;
  int evals;
  int density;
  int nc;
  bool firstsolved;
  bool random;
  float minx,miny,maxx,maxy;
	
  /*void calc_modularity(char*fn) {
  for(int i=0;i<nc;i++) {
  cout << "modularity niche " << i << endl;
  for(int j=0;j<5;j++) {	
  int ns=niches[order[i]].size();
  Organism *org = niches[order[i]][randint(0,ns-1)];
  modularity(org,fn);
  }
  }
  }*/	

  passive_niche(bool _r=false) {
    random=_r;
    density=30;
    niche_size=10;
    evals=250001;
    for(int i=0;i<MAX_NICHES;i++) explored[i]=false;
    for(int i=0;i<MAX_NICHES;i++) order[i]=false;
    nc=0;
    firstsolved=true;
  }


  void print_niches() {
    for(int x=0;x<density;x++) 
    {
      for(int y=0;y<density;y++)
      {
        cout << explored[x*density+y];
      }
      cout <<endl;
    }
  }

  int scale(int d,float val, float min,float max) {
    return (int)(d*(val-min)/((max+0.01f)-min));
  }

  int map_into_niche(Organism* o) {
    float x= o->noveltypoint->data[0][0];
    float y= o->noveltypoint->data[0][1];
    if (!random)
      return ((int)x)/10*300+((int)y)/10; //to match other simulations
    else
      return rand()%400;
    //return scale(density,x,minx,maxx)*density+scale(density,y,miny,maxy);
  }

  void insert_population(Population* pop) {
    vector<Organism*>::iterator curorg;
    for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
      insert_org((*curorg)); 
    }
  }

  void insert_org(Organism* org) {
    int target_niche = map_into_niche(org);
    if(target_niche<0)
      return; 
    int sz=niches[target_niche].size();
    if(sz==0) {
      explored[target_niche]=true;
      order[nc]=target_niche;
      nc++;
    }
    if(sz>=niche_size) {
      //return;
      remove_one_from_niche(target_niche);
    }
    niches[target_niche].push_back(org);
  }

  void remove_one_from_niche(int n) {
    //cout << "deleting from " << n << endl;

    int to_rem = randint(0,niches[n].size()-1);
    Organism* o = niches[n][to_rem];
    niches[n].erase(niches[n].begin()+to_rem);
    delete o;
  }
  int exploredcount() {
    int c=0;
    for(int i=0;i<MAX_NICHES;i++) if (explored[i]) c++;
    return  c;
  }
  void run_niche(Population* initpop) {
    //num evals
    envList[0]->get_range(minx,miny,maxx,maxy);
    insert_population(initpop);
    int e=0;
    int num_niches=density*density;
    while(e<evals) {
      cout << "evals " << e <<endl;
      cout << "explored " << exploredcount() << endl;
      vector<Organism*> children;
      //print_niches();
      int conn=0;
      int nodes=0;
      int count=0;
      for(int i=0;i<num_niches;i++) {
        if(niches[i].size()==0)
          continue;
        Genome *new_gene= new Genome(*niches[i][randint(0,niches[i].size()-1)]->gnome);
        //for(vector<Organism*>::iterator it=niches[i].begin();it!=niches[i].end();it++)
        //{
        //Genome *new_gene= new Genome(*(*it)->gnome);
        mutate_genome(new_gene,true);
        Organism* new_org= new Organism(0.0,new_gene,0);
        initpop->evaluate_organism(new_org);
        if(new_org->datarec->ToRec[3] > 0 && firstsolved) {
          cout << "solved " << e << endl;
          firstsolved=false;
        }
        nodes+=new_org->net->nodecount();
        conn+=new_org->net->linkcount();
        count++;
        children.push_back(new_org);
        e++;
        int upcnt=10000;
        if(e%upcnt==0) {
          char fn[100];
          sprintf(fn,"%s_modularity%d.dat",output_dir,e/upcnt);
          //calc_modularity(fn);
        }
        //}
      }
      cout << "avgnodes" << ((float)nodes)/count << endl;
      cout << "avgconns" << ((float)conn)/count << endl;
	
      for(vector<Organism*>::iterator it=children.begin();it!=children.end();it++)
        insert_org(*it);

    }
  }

};

  float contained_dist(float x,float y) {
    if(contained(x,y)) return 0.0;
    float xd=0;
    float yd=0;
    if(x>200)
      xd=(x-200);
    else if(x<0)
      xd= -x;
    if(y>200)
      yd=(y-200);
    else if(y<0)
      yd = -y;
    return (xd*xd+yd*yd);
  }

  bool contained(float x,float y)
  {
    Point p(x,y);
    if (x>200)
      return false;
    if (x<0)
      return false;
    if (y>200)
      return false;
    if (y<0)
      return false;

    return true;
    /*
    if(envList[0]->end.distance(p) < 400)
    {
    //cout <<"contained.." << endl;
    return true;
    }
    //cout <<"notcontained..." << endl;
    return false;
    */
  }

  void mutate_genome(Genome* new_genome,bool traits)
  {
    Network* net_analogue;
    double mut_power=NEAT::weight_mut_power;
    static double inno=1000;
    static int id=1000;
    new_genome->mutate_link_weights(mut_power,1.0,GAUSSIAN);
    if(traits) {
      vector<Innovation*> innos;
      if (randfloat()<NEAT::mutate_node_trait_prob) {
        //cout<<"mutate_node_trait"<<endl;
        new_genome->mutate_node_parameters(NEAT::time_const_mut_power,NEAT::time_const_mut_prob,
        NEAT::bias_mut_power,NEAT::bias_mut_prob);
      }

      if (randfloat()<NEAT::mutate_add_node_prob) 
        new_genome->mutate_add_node(innos,id,inno);
      else if (randfloat()<NEAT::mutate_add_link_prob) {
        //cout<<"mutate add link"<<endl;
        net_analogue=new_genome->genesis(0);
        new_genome->mutate_add_link(innos,inno,NEAT::newlink_tries);
        delete net_analogue;
      }


      if(randfloat()<0.5)
        new_genome->mutate_random_trait();
      if(randfloat()<0.2)
        new_genome->mutate_link_trait(1);
    }

    return;
  }

  /*void modularity(Organism* org,char* fn) {
  bool solution=false;
  fstream file;
  file.open(fn,ios::app|ios::out);
  cout <<"Modularity..." << endl;
  // file << "---" <<  " " << org->winner << endl;
  float minx,maxx,miny,maxy;
  envList[0]->get_range(minx,miny,maxx,maxy);
  double ox,oy,fit;
  int nodes;
  int connections;
  float mod=modularity_score(org->gnome);
  Genome *new_gene= new Genome(*org->gnome);
  Organism *new_org= new Organism(0.0,new_gene,0);
  noveltyitem* nov_item = maze_novelty_map(new_org);
  fit=nov_item->fitness;
  nodes=new_org->net->nodecount();
  connections=new_org->net->linkcount();
  ox=nov_item->data[0][0];
  oy=nov_item->data[0][1];
  if (nov_item->fitness>340) solution=true;

  //HOW IT WAS:
  //points[i*2]=(nov_item->data[0][0]-minx)/(maxx-minx);
  //points[i*2+1]=(nov_item->data[0][1]-miny)/(maxy-miny);
  delete new_org;
  delete nov_item;
  //file << endl;
  file << mod << " " << ox << " " << oy << " " << nodes << " " <<connections << " " << fit << " " << solution << endl;
  file.close();
  return;
  }*/

  float classify(vector<float>& results,vector<vector<float> >& data,Network* net,bool debug=false,bool real_val=false) {
    float correct=0;

    results.clear();
    for(int i=0;i<data.size();i++) {
      vector<float> line=data[i];
      float c_output = line[line.size()-1];
      double inputs[50];

     
      for(int j=0;j<line.size()-1;j++) {
        inputs[j]=line[j];
        if(debug)
          cout << inputs[j] << " ";
      }     


      net->flush();
      net->load_sensors(inputs);

      for (int z=0; z<10; z++)
        net->activate();

      float routput=net->outputs[0]->activation;
      float output=routput;
     
      if(debug)
        cout << output << " " << c_output << endl;
      
      if(!regression) {
        if (routput>0.5) output=1.0;
        else output=0.0;
      }
     

      //if (output==c_output)
      //	correct+=1;

      float error=c_output-output;
      error*=error;
      correct+=error;

      if (real_val) output=routput;
      results.push_back(output);
    }
    //if(debug)
    //  net->print_links_tofile("outy.dat");

    return 1.0 - (correct/data.size());

  }

  void accum_vect(vector<float>& acc,vector<float>& add) {
    vector<float>::iterator it1=acc.begin();
    vector<float>::iterator it2=add.begin();
    while(it1!=acc.end()) {
      //cout << (*it2) <<endl;
      (*it1)+=(*it2);
      //cout << (*it2) <<endl;
      it1++;
      it2++;
    }

  }

  void scale_vect(vector<float>& v,float factor) {
    vector<float>::iterator it=v.begin();
    while(it!=v.end()) {
      (*it)*=factor;
      it++;
    }
  }

  void precalc_outputs(vector<vector<float> > &outputs, vector<vector<float> > data, vector<Organism*> orgs) {

    int orgs_size = orgs.size();

    for(int i=0;i<orgs_size;i++) {
      vector<float> results;
      float perf=classify(results,data,orgs[i]->gnome->genesis(0));
      outputs.push_back(results);
    }

  }

  //todo:optimize
  float classify_ensemble_precalc(vector<float>& results,vector<vector<float> >& outputs, vector<vector <float> > data, vector<int> p,bool print_out=false) {

    int c_ind = data[0].size()-1;
    float err=0.0;
    float var=0.0;

    for(int i=0;i<data.size();i++) {
      float ans = data[i][c_ind];
      float accum = 0.0f; 

      vector<float> one_row;
      for(int j=0;j<p.size();j++) {
        float out = outputs[p[j]][i];
        one_row.push_back(out);
        accum+=out;
      }

      float prediction=accum/p.size();
      for(int j=0;j<p.size();j++) {
        float delta=one_row[j]-prediction;
        var+=delta*delta;
      }


      results.push_back(prediction);
      float delta=ans-prediction;
      err+= (1.0-delta*delta);
    }
    if(print_out)
      cout << "variance:" << var/data.size() << endl;
    return err/data.size();
  }

  float classify_ensemble(vector<float>& results,vector<vector<float> >& data, vector<Organism*> p,bool print_out=false) {

    vector<vector<float> > outputs;
    precalc_outputs(outputs,data,p);
    vector<int> seq;

    for(int i=0;i<p.size();i++)
      seq.push_back(i);

    return classify_ensemble_precalc(results,outputs,data,seq,print_out);


    vector<float> r_temp;
    vector<float> r_accum;

    for(int i=0;i<data.size();i++)
      r_accum.push_back(0.0);

    float weight_total=0.0;
 
    for(int i=0;i<p.size();i++) {
      float weight = 1.0; //p[i]->noveltypoint->fitness;
      //  if(weight<0.9 && !((weight_total==0.0 && i==(p.size()-1))))
      //  continue;
      //weight=1.0;
      //  cout << weight << endl;
      Network *newnet = p[i]->gnome->genesis(0);

      r_temp.clear();
      classify(r_temp,data,newnet,false,false);
      scale_vect(r_temp,weight);
      accum_vect(r_accum,r_temp);  

      weight_total+=weight;
      delete(newnet);
    }

    scale_vect(r_accum,1.0/weight_total);

    float c_index = data[0].size()-1;
    float correct=0;
 
    results.clear();
    float disagree=0.0f;
    for(int i=0;i<r_accum.size();i++) {

      float c_output=data[i][c_index];
      float output=r_accum[i];

      float t_disagree=1.0-output;
      if(output<t_disagree)
        t_disagree=output;
      disagree+=t_disagree;

      if(!regression) {
        if(output>0.5) output=1.0;
        else output=0.0;
      }
  
      //cout << output << endl;

      float error=c_output-output;
      error*=error;
      correct+=error;
      //if(output==c_output)
      // correct+=1;
      results.push_back(output);

    }
    if(print_out)
      cout << "disagreement: " << disagree/data.size() << endl;
    return 1.0- ((float)correct)/data.size();
  }


  void choose_ensemble(vector<vector<float> >& data,vector<Organism*> orgs,vector<Organism*>& ens) {
    int orgs_size=orgs.size();

    vector<int> ens_ind;
    vector< vector< float> > outputs;

    precalc_outputs(outputs,data,orgs);

    for(int i=0;i<10;i++) {
      int best_index=0;
      float best_perf=0;
      for(int j=0;j<orgs_size;j++) {
        vector<float> res;
        ens_ind.push_back(j);
        float perf=classify_ensemble_precalc(res,outputs,data,ens_ind);
        if(perf>best_perf) {
          best_perf=perf;
          best_index=j;
        }
        ens_ind.pop_back(); 
      }
      cout << i << " " << best_index << endl;
      ens_ind.push_back(best_index);
      ens.push_back(orgs[best_index]);
    }

  }


  noveltyitem* classifier_novelty_map(Organism *org,data_record* record) {
    static int best = 0;
    noveltyitem *new_item = new noveltyitem;
    Network* net = org->net;
    new_item->genotype=new Genome(*org->gnome);
    new_item->phenotype=new Network(*org->net);
    vector< vector<float> > gather;

    double fitness=0.0001;
    static float highest_fitness=0.0;

    new_item->viable=true;


    gather.clear();
 
    //todo: optimize
    vector<float> classifications;
    vector<float> classifications_gen;
    fitness= classify(classifications,classifier_train_data,net) + 0.0001;
    float generalization = classify(classifications_gen,classifier_valid_data,net);

    gather.push_back(classifications_gen);
  
    /*
    //measure novelty by which inputs the ANN is listening to
    vector<float> listening;
    vector<int> l_nodes;
    
    for(int i=0;i<40;i++) 
    listening.push_back(0.0);
    
    net->listening_nodes(l_nodes); 
    
    float val=1.0/l_nodes.size();
    for(int i=0;i<l_nodes.size();i++) {
    int node=l_nodes[i]-1;
    listening[node]=val;
    }
    gather.push_back(listening);
    */

    if (fitness>highest_fitness)
      highest_fitness=fitness;

    for (int i=0; i<gather.size(); i++)
      new_item->data.push_back(gather[i]);

    new_item->fitness=fitness;
    new_item->secondary=fitness;
    org->fitness=fitness;

    return new_item;
  }

  static int maxgens;

  population_state* create_classifier_popstate(char* outputdir,const char* classfile,int param,const char *genes, int gens,bool novelty) {

    maxgens=gens;
    float archive_thresh=3.0;
    noveltyarchive *archive= new noveltyarchive(archive_thresh,*maze_novelty_metric,true,push_back_size,minimal_criteria,true);

    //if doing multiobjective, turn off speciation, TODO:maybe turn off elitism
    if (NEAT::multiobjective) NEAT::speciation=false;

    Population *pop;

    Genome *start_genome;
    char curword[20];
    int id;

    ostringstream *fnamebuf;
    int gen;

    if (!seed_mode)
      strcpy(seed_name,genes);
    if(seed_mode)
      cout << "READING IN SEED" << endl;
    ifstream iFile(seed_name,ios::in);
 
    char trainfile[200];
    char testfile[200];
    char validfile[200];
    sprintf(trainfile,"%s_train.dat",classfile);
    sprintf(testfile,"%s_test.dat",classfile);
    sprintf(validfile,"%s_valid.dat",classfile);
     
    classifier_train_data = read_classifier_data(trainfile);
    classifier_test_data = read_classifier_data(testfile);
    classifier_valid_data = read_classifier_data(validfile);

    if (outputdir!=NULL) strcpy(output_dir,outputdir);
    cout<<"START GENERATIONAL MAZE EVOLUTION"<<endl;

    cout<<"Reading in the start genome"<<endl;
    //Read in the start Genome
    iFile>>curword;
    iFile>>id;
    cout<<"Reading in Genome id "<<id<<endl;
    start_genome=new Genome(id,iFile);
    iFile.close();

    cout<<"Start Genome: "<<start_genome<<endl;

    //Spawn the Population
    cout<<"Spawning Population off Genome"<<endl;
    cout << "Start genomes node: " << start_genome->nodes.size() << endl;
    if(!seed_mode) 
      pop=new Population(start_genome,NEAT::pop_size);
    else
      pop=new Population(start_genome,NEAT::pop_size,0.0);
    cout<<"Verifying Spawned Pop"<<endl;
    pop->verify();
   
    //set evaluator
    pop->set_evaluator(&classifier_novelty_map);
    pop->evaluate_all();
    delete start_genome;
    return new population_state(pop,novelty,archive);

  }

  Population *classifier_generational(char* output_dir,const char* classfile,int param, const char *genes, int gens, bool novelty) {
    char logname[100];
    char fname[100];
    sprintf(logname,"%s_log.txt",output_dir);
    logfile=new ofstream(logname);
    //pop->set_compatibility(&behavioral_compatibility);    
    population_state* p_state = create_classifier_popstate(output_dir,classfile,param,genes,gens,novelty);
    
    for (int gen=0; gen<=maxgens; gen++)  { //WAS 1000
      cout<<"Generation "<<gen<<endl;
      bool win = classifier_generational_epoch(p_state,gen);
      p_state->pop->epoch(gen);

      if (win)
      {
        break;
      }

    }

    sprintf(fname,"%s_final",output_dir);
    p_state->pop->print_to_file_by_species(fname);
    delete logfile;
    delete p_state;
    return NULL;
  }

  void destroy_organism(Organism* curorg) {
    curorg->fitness = SNUM/1000.0;
    curorg->noveltypoint->reset_behavior();
    curorg->destroy=true;
  }

  int classifier_success_processing(population_state* pstate) {
    static int gen=0; 
    double& best_fitness = pstate->best_fitness;
    double& best_secondary = pstate->best_secondary;

    vector<Organism*>::iterator curorg;
    Population* pop = pstate->pop;
    //Evaluate each organism on a test
    int indiv_counter=0;
 
    Organism* cur_champ=NULL;
    float high_fit=0.0;

    for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {

      if ((*curorg)->noveltypoint->fitness > high_fit) {
        cur_champ=*curorg;
        high_fit=((*curorg)->noveltypoint->fitness);
      }

      if ((*curorg)->noveltypoint->fitness > best_fitness)
      {
        best_fitness = (*curorg)->noveltypoint->fitness;
        cout << "NEW BEST " << best_fitness << endl;
      }

      if (!pstate->novelty)
        (*curorg)->fitness = (*curorg)->noveltypoint->fitness;
    }

    if(logfile!=NULL)
      (*logfile) << pstate->generation*NEAT::pop_size<< " " << best_fitness << " " << best_secondary << " " << time(0) << endl;
   
    //vector<Organism*> orgs;
    vector<Organism*>& orgs=pop->organisms;
    vector<Organism*> ensemble;
    gen++;

    vector<float> ens_results;
    //choose_ensemble(classifier_valid_data,orgs,ensemble);

    cout << "CHAMP TRAIN PERF: " << classify(ens_results,classifier_train_data,cur_champ->gnome->genesis(0)) << endl;
    cout << "CHAMP TEST PERF: " << classify(ens_results,classifier_test_data,cur_champ->gnome->genesis(0)) << endl;
    //    cout << "ENSEMBLE TRAIN PERF: "  << classify_ensemble(ens_results,classifier_train_data,ensemble,true) << endl;

    //    cout << "ENSEMBLE TEST PERF: "  << 
    //    classify_ensemble(ens_results,classifier_test_data,ensemble,true) << endl;

    return 0;
  }

  int maze_success_processing(population_state* pstate) {
    double& best_fitness = pstate->best_fitness;
    double& best_secondary = pstate->best_secondary;

    static bool win=false;
    static bool firstflag=false;
    static bool weakfirst=false;

    vector<Organism*>::iterator curorg;
    Population* pop = pstate->pop;
    //Evaluate each organism on a test
    int indiv_counter=0;
    for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {

      data_record* newrec = (*curorg)->datarec;
      if (!weakfirst && (newrec->ToRec[3]>=envList.size())) {
        weakfirst=true;
        char filename[100];
        cout << "Maze weakly solved by indiv# " << indiv_counter << endl;
        //disable quit for now
      }

	
      //write out the first individual to solve maze
      if (!firstflag && (newrec->ToRec[3]>=envList.size() && newrec->ToRec[4]>=envList.size())) {
        if ((*curorg)->noveltypoint->secondary >best_secondary) {
          best_secondary=(*curorg)->noveltypoint->secondary;
          cout << "NEW BEST SEC " << best_secondary << endl;

        }
      

        if( (*curorg)->noveltypoint->viable) {
          //cout << (*curorg)->noveltypoint->viable << endl;
		
          cout << "Maze solved by indiv# " << indiv_counter << endl;
          cout << newrec->ToRec[5] << endl;
          //break;
          (*curorg)->winner=true;
          win=true;

          if (fitness_measure==fitness_goal) {
            (*curorg)->winner=true;
            win=true;
            if ((*curorg)->noveltypoint->secondary >best_secondary) {
              best_secondary=(*curorg)->noveltypoint->secondary;
              cout << "NEW BEST SEC " << best_secondary << endl;

            }
          }
        }
      }
      if ((*curorg)->noveltypoint->fitness > best_fitness)
      {
        best_fitness = (*curorg)->noveltypoint->fitness;
        //cout << "NEW BEST " << best_fitness << endl;
      }

      indiv_counter++;
      if ((*curorg)->noveltypoint->viable && !pstate->mc_met)
        pstate->mc_met=true;
      else if (pstate->novelty && !(*curorg)->noveltypoint->viable && minimal_criteria && pstate->mc_met)
      {
        destroy_organism((*curorg));
      }

      if (!pstate->novelty)
        (*curorg)->fitness = (*curorg)->noveltypoint->fitness;
    }

    if(logfile!=NULL)
      (*logfile) << pstate->generation*NEAT::pop_size<< " " << best_fitness << " " << best_secondary << endl;

    if (win && !firstflag)
    {
      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        if ((*curorg)->winner) {
          cout<<"WINNER IS #"<<((*curorg)->gnome)->genome_id<<endl;
          char filename[100];
          sprintf(filename,"%s_%d_winner", output_dir,pstate->generation);
          (*curorg)->print_to_file(filename);
        }
      }
      firstflag=true;
    }
    if(win)
      return 1;
    return 0;
  }

  int maze_generational_epoch(population_state* pstate,int generation) {
    return 
      generalized_generational_epoch(pstate,generation,&maze_success_processing);
  }
  int classifier_generational_epoch(population_state* pstate,int generation) {
    return 
      generalized_generational_epoch(pstate,generation,&classifier_success_processing);
  }



  int generalized_generational_epoch(population_state* pstate,int generation,successfunc success_processing) {
    pstate->generation++;

    bool novelty = pstate->novelty;
    noveltyarchive& archive = *pstate->archive;
    data_rec& Record = pstate->Record;
    Population **pop2 = &pstate->pop;
    Population* pop= *pop2;
    vector<Organism*>::iterator curorg,deadorg;
    vector<Species*>::iterator curspecies;
    vector<Organism*>& measure_pop=pstate->measure_pop;
    
    cout << "#GENOMES:" << Genome::increment_count(0) << endl;
    cout << "#GENES:" << Gene::increment_count(0) << endl;
    cout << "ARCHIVESIZE: " << archive.get_set_size() << endl;

    int evolveupdate=100;

    if (NEAT::multiobjective) {  //merge and filter population	
      if(!novelty) NEAT::fitness_multiobjective=true;
      //if using alps-style aging
      if(pstate->max_age!=-1) 
      for (curorg=(measure_pop.begin());curorg!=measure_pop.end();++curorg) {
        (*curorg)->age++;	
        //if too old, delete
        if((*curorg)->age > pstate->max_age) {
          deadorg=curorg;
          if(pstate->promote!=NULL) {
            pstate->promote->measure_pop.push_back((*curorg));
          }
          else
            delete (*curorg);
          curorg=measure_pop.erase(deadorg);
          curorg--;
        }
      }

      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        measure_pop.push_back(new Organism(*(*curorg),true)); //TODO:maybe make a copy?
      }
        
      Genome* sg=pop->start_genome;
      evaluatorfunc ev=pop->evaluator; 
      delete pop;
      pop=new Population(measure_pop);
      pop->start_genome=sg;
      pop->set_evaluator(ev);
      *pop2= pop;
    }

    if (NEAT::evolvabilitytest)
      std::cout << "evol test still used...!!!!!!" << std::endl;

    int ret = success_processing(pstate);
    if(ret!=0)
      return 1;

    if (novelty)
    {

      archive.evaluate_population(pop,true);
      archive.evaluate_population(pop,false);

      pop->print_divtotal();


      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        if ( !(*curorg)->noveltypoint->viable && minimal_criteria)
        {
          (*curorg)->fitness = SNUM/1000.0;
          (*curorg)->noveltypoint->fitness = SNUM/1000.0;
          (*curorg)->noveltypoint->novelty = SNUM/1000.0;
        }
      }
      //cout << "ARCHIVE SIZE:" << archive.get_set_size() << endl;
      //cout << "THRESHOLD:" << archive.get_threshold() << endl;
      archive.end_of_gen_steady(pop);
    }
   
    if (NEAT::multiobjective) {
      archive.rank(pop->organisms);

      if (pop->organisms.size()>NEAT::pop_size) {
        //chop population down by half (maybe delete orgs that aren't used)
        int start=NEAT::pop_size; //measure_pop.size()/2;
        vector<Organism*>::iterator it;
        for (it=pop->organisms.begin()+start; it!=pop->organisms.end(); it++) {
          (*it)->species->remove_org(*it);
          delete (*it);
        }
        pop->organisms.erase(pop->organisms.begin()+start,pop->organisms.end());
      }
    }
    

	
#ifdef PLOT_ON11
    if(true) {
      vector<float> x,y,z;
      pop->gather_objectives(&x,&y,&z);
      front_plot.plot_data(x,y,"p","Pareto front");
      //best_fits.push_back(pstate->best_fitness);
      //fitness_plot.plot_data(best_fits,"lines","Fitness");

      /*
      vector<float> xc;
      vector<float> yc;
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {
      int sz=(*curorg)->noveltypoint->data[0].size();
      //xc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
      //yc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
      if((*curorg)->noveltypoint->viable) {
      xc.push_back((*curorg)->noveltypoint->data[0][sz-3]);
      yc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
      }
      }
      behavior_plot.plot_data(xc,yc);

      }
      */
      vector<float> xc;
      vector<float> yc;
      float coltot=0.0;
      for (curorg = (pop->organisms).begin(); curorg != pop->organisms.end(); ++curorg) {

        coltot+=(*curorg)->datarec->ToRec[5];
        int sz=(*curorg)->noveltypoint->data[0].size();
        if((*curorg)->datarec->ToRec[5]>-5) {
          xc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
          xc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
        }
        else {
          yc.push_back((*curorg)->noveltypoint->data[0][sz-2]);
          yc.push_back((*curorg)->noveltypoint->data[0][sz-1]);
        }
      }
      cout << "COLTOT: " << coltot << endl;
      vector<vector <float> > blah;
      blah.push_back(xc);
      blah.push_back(yc);
      behavior_plot.plot_data_2d(blah);

    }
#endif

 
    char fn[100];
    sprintf(fn,"%sdist%d",output_dir,generation);
    if (NEAT::printdist)
      pop->print_distribution(fn);
    
    for (curspecies=(pop->species).begin(); curspecies!=(pop->species).end(); ++curspecies) {
      (*curspecies)->compute_average_fitness();
      (*curspecies)->compute_max_fitness();
    }

    //writing out stuff
    if ((generation+1)%NEAT::print_every == 0 )
    {
      char filename[100];
      sprintf(filename,"%s_record.dat",output_dir);
      char fname[100];
      sprintf(fname,"%s_archive.dat",output_dir);
      archive.Serialize(fname);
      sprintf(fname,"%sgen_%d",output_dir,generation);
      pop->print_to_file_by_species(fname);
        
      sprintf(fname,"%sfittest_%d",output_dir,generation);
      archive.serialize_fittest(fname);
    }

    if (NEAT::multiobjective) {
      for (curorg=measure_pop.begin(); curorg!=measure_pop.end(); curorg++) delete (*curorg);
      //clear the old population
      measure_pop.clear();
      if (generation!=maxgens)
      for (curorg=(pop->organisms).begin(); curorg!=(pop->organisms).end(); ++curorg) {
        measure_pop.push_back(new Organism(*(*curorg),true));
      }
    }

    //Create the next generation
    pop->print_avg_age();

    return 0;
  }

