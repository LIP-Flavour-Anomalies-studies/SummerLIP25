#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

void ROOTvar(){

    // Create output directory if it doesn't exist
    fs::create_directories("Machine_Learning/Data_Application/ROOT");

    // --- Open File and Get Tree ---
    TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TTree *t_data = (TTree*)f_data->Get("ntuple");

    TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");

    // --- Output file and tree ---
    TFile *f_data_sel = new TFile("Machine_Learning/Data_Application/ROOT/data_selected.root", "RECREATE");
    TTree *t_data_sel = new TTree("Tdata", "Tree with selected variables from experimental data");

    TFile *f_mc_sel = new TFile("Machine_Learning/Data_Application/ROOT/mc_selected.root", "RECREATE");
    TTree *t_mc_sel = new TTree("Tdata", "Tree with selected variables from MC");

    // --- Storage for input/output variable values ---
	map<string, double> vars_data;
    map<string, double> vars_data_sel;
    map<string, double> vars_mc;
    map<string, double> vars_mc_sel;

    // --- Variable names ---
	vector<string> variables = {
        "bMass", "bMassE", "bBarMass", "bBarMassE", "bVtxCL", "bPt", "bPhi", "bEta",
        "kstMass", "kstMassE", "kstBarMass", "kstBarMassE", "kstPt", "kstPhi", "kstEta",
        "mumuMass", "mumuMassE", "mumuPt", "mumuPhi", "mumuEta",
        "kstTrkmPt", "kstTrkmPhi", "kstTrkmEta", "kstTrkmDCABS", "kstTrkmDCABSE",
        "kstTrkpPt", "kstTrkpPhi",  "kstTrkpEta", "kstTrkpDCABS", "kstTrkpDCABSE",
        "mumPt", "mumPhi", "mumEta", 
        "mupPt", "mupPhi", "mupEta",
        "tagB0", "bCosAlphaBS", "bLBS", "bLBSE", "bDCABS", "bDCABSE"
    };

    // variables to upload in files 
	vector<string> var_files = {
        "bTMass", "bVtxCL", "bPt", "bPhi", "bEta",
        "kstTMass", "kstPt", "kstPhi", "kstEta",
        "mumuMass", "mumuPt", "mumuPhi", "mumuEta",
        "kstTrkmPt", "kstTrkmPhi", "kstTrkmEta", "kstTrkmDCABS",
        "kstTrkpPt", "kstTrkpPhi",  "kstTrkpEta", "kstTrkpDCABS",
        "mumPt", "mumPhi", "mumEta", 
        "mupPt", "mupPhi", "mupEta",
        "bCosAlphaBS", "bLBS", "bDCABS",
        "muLeadingPt", "muTrailingPt",
        "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs",
        "kstTrkpPtR", "kstTrkmPtR", "muTrailingPtR", "muLeadingPtR",
        "mumuPtR", "kstPtR"
	};

    // newly created variables (not previously in the file)
    vector<string> var_new = {
        "bTMass", "kstTMass", 
        "muLeadingPt", "muTrailingPt",
        "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs",
        "kstTrkpPtR", "kstTrkmPtR", "muTrailingPtR", "muLeadingPtR",
        "mumuPtR", "kstPtR"
    };

    // variables for matching
    vector<string> var_match = {
        "truthMatchMum", "truthMatchMup", "truthMatchTrkm", "truthMatchTrkp"
    };

    // --- Allocate variables and set branch addreses ---
    for (const auto &name : variables){
        vars_data[name] = 0;
        vars_mc[name] = 0;
        t_data->SetBranchAddress(name.c_str(), &vars_data[name]);
		t_mc->SetBranchAddress(name.c_str(), &vars_mc[name]);
    }

    for (const auto &name : var_match){
        t_mc->SetBranchAddress(name.c_str(), &vars_mc[name]);
    }
    
    // --- Output Variables ---
    for (const auto &name : var_files){
        vars_data_sel[name] = 0;
        vars_mc_sel[name] = 0;
        t_data_sel->Branch(name.c_str(), &vars_data_sel[name]);
        t_mc_sel->Branch(name.c_str(), &vars_mc_sel[name]);
    }

    double mmin = 5;
    double mmax = 5.6;

    // --- Fill data file ---
    cout << "Looping over data..." << endl;
    Long64_t nEntries_data = t_data->GetEntries();

    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);    

        double mass_b = (vars_data["tagB0"] == 1) ? vars_data["bMass"] : vars_data["bBarMass"];
		if (mass_b < mmin || mass_b > mmax) continue;

        double mass_kst = (vars_data["tagB0"] == 1) ? vars_data["kstMass"] : vars_data["kstBarMass"];

        double muLeadingPt = max(vars_data["mupPt"], vars_data["mumPt"]);
        double muTrailingPt = min(vars_data["mupPt"], vars_data["mumPt"]);

        
        for (const auto &name : var_files){
            // Fill already existent variables
            if (find(var_new.begin(), var_new.end(), name) == var_new.end()){
                vars_data_sel[name] = vars_data[name];
            }
        }
        
        // Fill newly created variables
        vars_data_sel["bTMass"] = mass_b;
        vars_data_sel["kstTMass"] = mass_kst;
        vars_data_sel["muTrailingPt"] = muTrailingPt;
        vars_data_sel["muLeadingPt"] = muLeadingPt;

        // significance variables
        if (vars_data["bLBSE"] != 0){
            vars_data_sel["bLBSs"] = vars_data["bLBS"] / vars_data["bLBSE"];
        } 

        if (vars_data["bDCABSE"] != 0){
            vars_data_sel["bDCABSs"] = vars_data["bDCABS"] / vars_data["bDCABSE"];
        } 

        if (vars_data["kstTrkmDCABSE"] != 0){
            vars_data_sel["kstTrkmDCABSs"] = vars_data["kstTrkmDCABS"] / vars_data["kstTrkmDCABSE"];
        } 

        if (vars_data["kstTrkpDCABSE"] != 0){
            vars_data_sel["kstTrkpDCABSs"] = vars_data["kstTrkpDCABS"] / vars_data["kstTrkpDCABSE"];
        } 

        // relative pt 
        if (vars_data["bPt"] != 0){
            vars_data_sel["kstTrkpPtR"] = vars_data["kstTrkpPt"] / vars_data["bPt"];
            vars_data_sel["kstTrkmPtR"] = vars_data["kstTrkmPt"] / vars_data["bPt"];
            vars_data_sel["kstTrkpPtR"] = vars_data["kstTrkpPt"] / vars_data["bPt"];
            vars_data_sel["muLeadingPtR"] = vars_data["muLeadingPt"] / vars_data["bPt"];
            vars_data_sel["muTrailingPtR"] = vars_data["muTrailingPt"] / vars_data["bPt"];
            vars_data_sel["mumuPtR"] = vars_data["mumuPt"] / vars_data["bPt"];
            vars_data_sel["kstPtR"] = vars_data["kstPt"] / vars_data["bPt"];
        } 

        t_data_sel->Fill();
    }  
    cout << "Finished processing data" << endl;


    // --- Fill monte carlo file --- 
    cout << "Looping over MC..." << endl;
    Long64_t nEntries_mc = t_mc->GetEntries();
    for(Long64_t i = 0; i < nEntries_mc; i++){    
        t_mc->GetEntry(i);

        // Truth matching cuts first to isolate signal
        bool match = true;
        for (const auto &name : var_match){
            if (vars_mc[name] == 0) match = false;
        }
        if (!match) continue;

        double muLeadingPt = max(vars_mc["mupPt"], vars_mc["mumPt"]);
        double muTrailingPt = min(vars_mc["mupPt"], vars_mc["mumPt"]);

        double mass_b = (vars_mc["tagB0"] == 1) ? vars_mc["bMass"] : vars_mc["bBarMass"];
		if (mass_b < mmin || mass_b > mmax) continue;

        double mass_kst = (vars_mc["tagB0"] == 1) ? vars_mc["kstMass"] : vars_mc["kstBarMass"];

        
        for (const auto &name : var_files){
            if (find(var_new.begin(), var_new.end(), name) == var_new.end()){
                vars_mc_sel[name] = vars_mc[name];
            }
        }
        
        vars_mc_sel["bTMass"] = mass_b;
        vars_mc_sel["kstTMass"] = mass_kst;
        vars_mc_sel["muTrailingPt"] = muTrailingPt;
        vars_mc_sel["muLeadingPt"] = muLeadingPt;

        if (vars_mc["bLBSE"] != 0){
            vars_mc_sel["bLBSs"] = vars_mc["bLBS"] / vars_mc["bLBSE"];
        } 

        if (vars_mc["bDCABSE"] != 0){
            vars_mc_sel["bDCABSs"] = vars_mc["bDCABS"] / vars_mc["bDCABSE"];
        } 

        if (vars_mc["kstTrkmDCABSE"] != 0){
            vars_mc_sel["kstTrkmDCABSs"] = vars_mc["kstTrkmDCABS"] / vars_mc["kstTrkmDCABSE"];
        } 

        if (vars_mc["kstTrkpDCABSE"] != 0){
            vars_mc_sel["kstTrkpDCABSs"] = vars_mc["kstTrkpDCABS"] / vars_mc["kstTrkpDCABSE"];
        } 

        if (vars_mc["bPt"] != 0){
            vars_mc_sel["kstTrkpPtR"] = vars_mc["kstTrkpPt"] / vars_mc["bPt"];
            vars_mc_sel["kstTrkmPtR"] = vars_mc["kstTrkmPt"] / vars_mc["bPt"];
            vars_mc_sel["kstTrkpPtR"] = vars_mc["kstTrkpPt"] / vars_mc["bPt"];
            vars_mc_sel["muLeadingPtR"] = vars_mc["muLeadingPt"] / vars_mc["bPt"];
            vars_mc_sel["muTrailingPtR"] = vars_mc["muTrailingPt"] / vars_mc["bPt"];
            vars_mc_sel["mumuPtR"] = vars_mc["mumuPt"] / vars_mc["bPt"];
            vars_mc_sel["kstPtR"] = vars_mc["kstPt"] / vars_mc["bPt"];
        } 

        t_mc_sel->Fill();
        
    }
    cout << "Finished processing MC" << endl;


    f_data->Close();
    f_mc->Close();

    f_data_sel->cd();
    t_data_sel->Write();
    f_data_sel->Close();

    f_mc_sel->cd();
    t_mc_sel->Write();
    f_mc_sel->Close();

}