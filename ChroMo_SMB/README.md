# IE SMB-sim

Application for integration of idustrial SMB chromatography processes into Industrial Edge.

---


- [IE SMB-sim](#ie-smb-sim)
  - [Description](#description)
    - [Overview](#overview)
  - [Requirements](#requirements)
    - [Used components](#used-components)
    - [Further requirements](#further-requirements)
  - [Installation](#installation)
    - [Configuring the Industrial Edge Databus](#configuring-the-industrial-edge-databus)
    - [Configuring the SIMATIC S7 Connector](#configuring-the-simatic-s7-connector)
    - [Configuring the Data Service](#configuring-the-data-service)
  - [Usage](#usage)
    - [Dependencies](#dependencies)
  - [Documentation](#documentation)

## Description

This application can simulate chromatography process based on input parameters. It also can monitor and control real chromatography process and thus can be used for model predictive control.

Application is desinged with specific PLC tag layout in mind.

Application is in development.

### Overview

Application example can be downloaded on [GitHub repository](https://github.com/svoboad3/SMBSimulator).

Use Dockerfile and docker-compose.yml with Industrial Edge App Publisher to import application to Industrial Edge Management.

## Requirements

### Used components

This application was created and tested using these components

- Industrial Edge App Publisher V1.9.5
- Docker Engine 20.10.21
- Docker Compose V2.4
- Industrial Edge Virtual Device V1.12.0-3-a
- IE Databus Configurator V2.0.0-5
- IE Databus V2.0.0-4
- IE Common Connector Configurator V1.8.2-3
- IE SIMATIC S7 Connector V1.8.1-7
- IE Data Service V1.5.0
- IE Management System V1.1.0-48

### Further requirements

- IE Device is onboarded to a IE Management
- IE Databus Configurator is deployed to the IE Management
- IE Databus is deployed to the IE Device

## Installation

1. Pull Git project from this repository.
2. Create docker image using `docker build -t "ie-smb-sim" .`.
3. Create application using IE App Publisher and docker-compose.yml file.
4. Upload it to IE Management.
5. Install it to IE Device.

Please refer to [Uploading App to IEM](https://github.com/industrial-edge/upload-app-to-industrial-edge-management) on how to upload the app to the IEM.

### Configuring the Industrial Edge Databus

IE SMB-sim application requires Industrial Edge Databus application to be installed and configured.

- In the Industrial Edge Management Web interface, click on "Data Connections" and select the Databus
- Select the corresponding Industrial Edge Device and click "Launch"
- Create a new user with the username and password defined as "edge" and "edge"
- Create the topic "ie/#" and give the user publish and subscribe permission
- Deploy the databus configuration and wait for the job to be finished successfully

### Configuring the SIMATIC S7 Connector

IE SMB-sim application requires SIMATIC S7 Connector application to be installed and configured.

- In the Industrial Edge Management Web interface, click on "Data Connections" and select the Databus
- Select the corresponding Industrial Edge Device and click "Launch"
- Import configuration file IEVD2.json
- Deploy the tags to Industrial Edge Device
- Start Project

### Configuring the Data Service

IE SMB-sim application requires Data Service application to be installed and configured.

- In the Industrial Edge Device Web interface, click on "Apps" and select the Data Service
- Select "Connectors" and activate predefined SIMATIC S7 Connector
- In "Assets & Connectivity" select Add multiple variables and select all variables from SIMATIC S7 Connector and click "Save"

## Usage

Application page can be accessed through IED in Apps tab.

First, login is required with same credentials as login to IED.

Then you can choose between offline and online simulation. Offline being just the simulation until specified time. Online running alongside real SMB process.

Next define SMB station by creating columns in specific zones. After there is at least one column in each zone, you can continue.

Next define separation mixture by adding components.

Next step in offline simulation is to define flow rates and model calculation parameters.

Next step in online simulation is to map tags to simulation parameters and define time and diferences.

After that you can launch simulation.

### Dependencies

In order for this application to run properly on Industrial Edge Device (IED), the following applications must be installed and configured on the IED

- Databus application
- IE Data Service application
- SIMATIC S7 Connector application

## Documentation
  
- You can find further documentation and help in the following links
  - [Industrial Edge Hub](https://iehub.eu1.edge.siemens.cloud/#/documentation)
  - [Industrial Edge Forum](https://www.siemens.com/industrial-edge-forum)
  - [Industrial Edge landing page](https://new.siemens.com/global/en/products/automation/topic-areas/industrial-edge/simatic-edge.html)
  - [Industrial Edge GitHub page](https://github.com/industrial-edge)
