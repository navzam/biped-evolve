#include <iostream>
#include "ode/ode.h"
#include <drawstuff/drawstuff.h>

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawLine dsDrawLineD
#endif

static dWorldID world;
static dBodyID sphere, box, capsule, cylinder;
const dReal MASS = 1.0;
const dReal RADIUS = 0.25;
const dReal LENGTH = 1.0;
const dReal DENSITY = 5.0;
const dReal SIDES[3] = {0.5, 0.5, 1.0};

static void start()
{
  std::cout << "Start" << std::endl;
  
  // Set camera position/direction
  static float xyz[3] = {5.0, 3.0, 0.5};
  static float hpr[3] = {-180.0, 0.0, 0.0};
  dsSetViewpoint(xyz, hpr);
}

// Simulation loop
static void simLoop(int pause)
{
  std::cout << "Sim loop" << std::endl;
  
  // Simulation time step
  dWorldStep(world, 0.05);
  
  // Draw a sphere
  const dReal *pos1 = dBodyGetPosition(sphere);
  const dReal *rot1 = dBodyGetRotation(sphere);
  dsSetSphereQuality(3);
  dsSetColor(1.0, 0.0, 0.0);
  dsDrawSphere(pos1, rot1, RADIUS);
  
  // Draw a box
  const dReal *pos2 = dBodyGetPosition(box);
  const dReal *rot2 = dBodyGetRotation(box);
  dsSetColorAlpha(0.0, 0.0, 1.0, 1.0);
  dsDrawBox(pos2, rot2, SIDES);
  
  // Draw a cylinder
  const dReal *pos3 = dBodyGetPosition(cylinder);
  const dReal *rot3 = dBodyGetRotation(cylinder);
  dsSetColorAlpha(0.0, 1.0, 0.0, 1.0);
  dsDrawCylinder(pos3, rot3, LENGTH, RADIUS);
}

int main(int argc, char **argv)
{  
  // For drawstuff
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &start;
  fn.step = &simLoop;
  fn.command = NULL;
  fn.stop = NULL;
  fn.path_to_textures = "../textures";
  
  // Init world
  dInitODE();
  world = dWorldCreate();
  dWorldSetGravity(world, 0, 0, -0.01);
  
  // Create mass for all objects
  dMass mass;
  dMassSetZero(&mass);
  
  // Create a sphere
  sphere = dBodyCreate(world);
  dMassSetSphere(&mass, DENSITY, RADIUS);
  dBodySetMass(sphere, &mass);
  dBodySetPosition(sphere, 0.0, 1.0, 1.0);
  
  // Create a box
  box = dBodyCreate(world);
  dMassSetBox(&mass, DENSITY, SIDES[0], SIDES[1], SIDES[2]);
  dBodySetMass(box, &mass);
  dBodySetPosition(box, 0.0, 2.0, 1.0);
  
  // Create a cylinder
  cylinder = dBodyCreate(world);
  dMassSetCylinder(&mass, DENSITY, 3, RADIUS, LENGTH);
  dBodySetMass(cylinder, &mass);
  dBodySetPosition(cylinder, 0.0, 3.0, 1.0);
  
  // Run simulation loop
  const int WINDOW_WIDTH = 640;
  const int WINDOW_HEIGHT = 480;
  dsSimulationLoop(argc, argv, WINDOW_WIDTH, WINDOW_HEIGHT, &fn);
  
  // Cleanup
  dWorldDestroy(world);
  dCloseODE();
  
  return 0;
}