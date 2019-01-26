# Solar Escape
AI solar system escape satellite. 

## Overview
The aim of this project is to build an AI agent for a theoretical satellite that will use information about nearby celestial bodies to escape its solar system by using them for gravitational assist maneuvers.
The agent's performance metric will be made up of the speed with which it exits the solar system, the amount of fuel it used to do so, and the time it took to exit.
The end result will consist of a solar system simulator and an agent (i.e., satellite) intelligent enough to escape that system, using gravity assist.
The simulation will be 2D, and the paths of the planets' orbits will be drawn as curved lines, and the agent will be simply represented as a dot.

## Goal
Using [Q-learning](https://en.wikipedia.org/wiki/Q-learning) to teach an agent to use gravital assist to escape a simulated solar system's gravitational pull.


## Built With
* Python3

## Dependencies
* OrbitalPy

## Get Started
make sure you have ``pip`` installed, then run 
``pip install orbitalpy``
