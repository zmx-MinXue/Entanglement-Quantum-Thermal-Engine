# Overview
The focus of this repository is a 2×d dimension qubit–oscillator quantum thermal engine, with qubit coupling to a hot bosonic bath, and oscillator coupling to a cold bosonic bath. 

Upto now, the code consists of: 
- Master equation construction (local / global)
- Steady-state negativity optimization (local / global)

The repository consists of three main parts:
- src/ — master equation construction + other utilities
- scripts/ — negativity optimization 
- results/ — data + data processing scripts

# Something to note: 
The maximum point of global 2xd system negativity, is actually not valid under the assumptions for global master equation.