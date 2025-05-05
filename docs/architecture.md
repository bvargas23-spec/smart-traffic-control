# System Architecture

This document outlines the architecture of the Smart Traffic Control System.

## Overview

The Smart Traffic Control System is designed with a modular, layered architecture that separates concerns and allows for flexible deployment across different environments, from simulation to physical traffic light control.

## System Layers

The system consists of five primary layers:

1. **Data Acquisition Layer**: Interfaces with external data sources
2. **Data Processing Layer**: Transforms raw data into actionable information
3. **Decision Layer**: Applies algorithms to determine optimal traffic signal timings
4. **Control Layer**: Interfaces with physical hardware or simulation environments
5. **Monitoring Layer**: Provides visualization and monitoring capabilities

## Component Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         Monitoring Layer                            │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ Web Dashboard   │  │ Traffic Visualizer │  │ Alert System   │   │
│  └─────────────────┘  └────────────────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                               ▲
                               │
┌────────────────────────────────────────────────────────────────────┐
│                         Decision Layer                              │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ Timing Optimizer│  │ Traffic Predictor  │  │ Pattern Detector│   │
│  └─────────────────┘  └────────────────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                               ▲
                               │
┌────────────────────────────────────────────────────────────────────┐
│                      Data Processing Layer                          │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ Data Processor  │  │ Traffic Analyzer   │  │ Historical DB   │   │
│  └─────────────────┘  └────────────────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                               ▲
                               │
┌────────────────────────────────────────────────────────────────────┐
│                     Data Acquisition Layer                          │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ TomTom Client   │  │ Weather Client     │  │ Sensor Client   │   │
│  └─────────────────┘  └────────────────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                               ▲
                               │
┌────────────────────────────────────────────────────────────────────┐
│                         Control Layer                               │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────────┐   │
│  │ Signal Controller│ │ BeagleBone Interface│ │ MQTT Client    │   │
│  └─────────────────┘  └────────────────────┘  └────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Data Acquisition Layer

- **TomTom Client** (`tomtom_client.py`): Interfaces with TomTom's traffic API to retrieve real-time traffic data
- **Weather Client**: Optional component to retrieve weather data that might affect traffic patterns
- **Sensor Client**: Interfaces with local sensors (if available) for supplementary traffic data

### Data Processing Layer

- **Data Processor** (`data_processor.py`): Converts raw API data into structured traffic information
- **Traffic Analyzer** (`traffic_analyzer.py`): Analyzes traffic patterns and identifies anomalies
- **Historical Database**: Stores historical traffic data for trend analysis and prediction

### Decision Layer

- **Timing Optimizer**: Calculates optimal signal timing based on current conditions
- **Traffic Predictor**: Predicts near-future traffic conditions to enable proactive signal control
- **Pattern Detector**: Identifies recurring traffic patterns for optimization

### Control Layer

- **Signal Controller**: Core logic for controlling traffic signals
- **BeagleBone Interface**: Hardware abstraction for BeagleBone Black
- **MQTT Client**: Enables communication between distributed system components

### Monitoring Layer

- **Web Dashboard**: Provides a user interface for system monitoring and configuration
- **Traffic Visualizer**: Creates visual representations of traffic conditions
- **Alert System**: Generates alerts for anomalous traffic conditions

## Communication Flow

1. The Data Acquisition Layer retrieves traffic data from TomTom API
2. The Data Processing Layer transforms this data into actionable metrics
3. The Decision Layer applies algorithms to determine optimal signal timing
4. The Control Layer implements these timing decisions on physical hardware
5. The Monitoring Layer displays system status and traffic conditions

## Hardware Integration

The system integrates with BeagleBone Black hardware via GPIO pins:

```python
# Example pin configuration for traffic signals
traffic_light_pins = {
    "NORTH": {"red": "P8_7", "yellow": "P8_8", "green": "P8_9"},
    "SOUTH": {"red": "P8_10", "yellow": "P8_11", "green": "P8_12"},
    "EAST": {"red": "P8_13", "yellow": "P8_14", "green": "P8_15"},
    "WEST": {"red": "P8_16", "yellow": "P8_17", "green": "P8_18"}
}
```

## Deployment Architecture

The system can be deployed in three configurations:

1. **Simulation Mode**: All components run in simulation without physical hardware
2. **Development Mode**: Components run on a development machine with mock hardware interfaces
3. **Production Mode**: Full deployment on BeagleBone Black with physical traffic light connections

## Scalability

The system is designed to scale in several dimensions:

- **Multiple Intersections**: The architecture supports coordination between multiple intersections
- **Additional Data Sources**: New data sources can be added to the Data Acquisition Layer
- **Enhanced Algorithms**: The Decision Layer can be extended with more sophisticated algorithms

## Security Considerations

- **API Key Management**: Sensitive API keys are stored in environment variables
- **Network Security**: When deployed, the BeagleBone should use secured network connections
- **Physical Security**: Physical access to the BeagleBone should be restricted

## Error Handling & Resilience

The system implements several resilience strategies:

- **Graceful Degradation**: If external data sources are unavailable, the system falls back to default timing plans
- **Error Recovery**: The Control Layer includes mechanisms to recover from hardware errors
- **Watchdog Processes**: Monitoring processes ensure system health and can restart components if necessary

## Future Extensions

The architecture is designed to accommodate future enhancements:

- **Machine Learning Integration**: More sophisticated traffic prediction using ML models
- **Vehicle-to-Infrastructure Communication**: Direct communication with connected vehicles
- **Multi-Intersection Coordination**: Coordinated control of traffic signals across multiple intersections