#include <Python.h>
#include <stdio.h>
#include <math.h>

#define GRAVITY_CONST 6.67428 * pow(10.0, -11.0)
#define C 299.792458

main(int argc, char **argv) {
    // Initialize the python interpreter
    Py_Initialize();
    // Initialize our module
    //init_calculations();
    // Exit the interpreter
    Py_Exit(0);
}

// Calculate Force
// F = (G * M_a * M_b) / distance(a, b)^2
static PyObject *calculateForce(PyObject *self, PyObject* args) {
    double force, distance;
    double dx, dy, ma, ra, mb, rb;

    if (!PyArg_ParseTuple(args, "dddddd", &dx, &dy, &ma, &ra, &mb, &rb))
        return NULL;

    // dx = bx - ax;
    // dy = by - ay;
    distance = sqrt(dx * dx + dy * dy);
    if(distance < (ra + rb)+10) distance = ra + rb; 

    ma = ma * 10000000000;
	mb = mb * 10000000000;
    distance = distance * 100000 ;

    force = (GRAVITY_CONST * ma * mb) / distance;

    return PyFloat_FromDouble( force );
}

// Calculate distance between two bodies a and b
// D = sqrt( (a_x - b_x)^2 + (a_y - b_y)^2 )
static PyObject *calculateDist(PyObject *self, PyObject* args) {
    double dx, dy, ra, rb, distance;
    if (!PyArg_ParseTuple(args, "dddd", &dx, &dy, &ra, &rb))
        return NULL;
    distance = sqrt(dx * dx + dy * dy);
    if(distance < (ra + rb)) distance = ra + rb; 

    return PyFloat_FromDouble(distance);
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
    //return
}

static PyMethodDef calculationMethods[] = {
	{"force",calculateForce, METH_VARARGS, "Return the Force of interaction between 2 bodies - (a_x,a_y,a_radius,a_mass,b_x,b_y,b_radius, b_mass)"},
	{"dist",calculateDist, METH_VARARGS, "Return the distance between 2 bodies - (a_x,a_y,a_radius,b_x,b_y,b_radius)"},
	//{"relativistic_mass",pyplanets_calculations_relativistic_mass,METH_VARARGS, "Return the relativistic mass of a body."},
	//{"collision",pyplanets_calculations_collision,METH_VARARGS, "Return the new location and speed after the collision"},
	{NULL, NULL}
};

static struct PyModuleDef calculations = 
{
    PyModuleDef_HEAD_INIT,
    "calculations",
    "",
    -1,
    calculationMethods
};

PyMODINIT_FUNC PyInit_calculations(void)
{
    return PyModule_Create(&calculations);
}
