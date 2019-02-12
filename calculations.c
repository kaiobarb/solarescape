#include <stdio.h>
#include <math.h>
#include <Python.h>

#define GRAVITY_CONST 6.67428 * pow(10.0, -11.0)
#define C 299.792458

void init_calculations(void);

main(int argc, char **argv) {
    // Initialize the python interpreter
    Py_Initialize();
    // Initialize our module
    init_calculations();
    // Exit the interpreter
    Py_Exit(0);
}

// Calculate Force
// F = (G * M_a * M_b) / distance(a, b)^2
static PyObject *calculateForce(PyObject *self, PyObject* args) {
    
}

// Calculate distance between two bodies a and b
// D = sqrt( (a_x - b_x)^2 + (a_y - b_y)^2 )
static PyObject *calculateDist(PyObject *self, PyObject* args) {

}

// Calculate relativistic mass
// From Wikipedia: "The measurable inertia and gravitational attraction of a 
// body in a given frame of reference is determined by its relativistic 
// mass, not merely its rest mass. For example, light has zero rest mass but
// contributes to the inertia (and weight in a gravitational field) of any 
// system containing it. "
// 
// RM = m_a / sqrt( 1 - ( ( sqrt(speed_x^2 + speed_y^2) )^2 ) / ( C^2 ) ); 
// where C = 299.792458 (speed of light)
static PyObject *calculateRelativeMass(PyObject * self, PyObject* args) {
    
}

static PyMethodDef calculationMethods[] = {
	{"forc",calculateForce, METH_VARARGS, "Return the Force of interaction between 2 bodies - (a_x,a_y,a_radio,a_mass,b_x,b_y,b_radio,b_mass)"},
	{"dist",calculateDist, METH_VARARGS, "Return the distance between 2 bodies - (a_x,a_y,a_radio,b_x,b_y,b_radio)"},
	//{"relativistic_mass",pyplanets_calculations_relativistic_mass,METH_VARARGS, "Return the relativistic mass of a body."},
	//{"collision",pyplanets_calculations_collision,METH_VARARGS, "Return the new location and speed after the collision"},
	{NULL, NULL}
};

void initCalculations(void){
    PyImport_AddModule("calculations");
    Py_InitModule("calculations", calculationMethods);
}
