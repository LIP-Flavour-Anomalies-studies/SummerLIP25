#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;

void ntuple(){

	gStyle->SetOptStat(0);

	TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	
	TTree *t_data = (TTree*)f_data->Get("ntuple");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");

	// Storage for variable values
	map<string, double> vars_data;
	map<string, double> vars_mc;

	// Variable names
	vector<string> variables = {
        "bMass", "bMassE", "bBarMass", "bBarMassE", "bVtxCL", "kstMass", "kstMassE",
        "kstBarMass", "kstBarMassE", "mumuMass", "mumuMassE", "bPt", "kstPt", "mumuPt",
        "mumPt", "mupPt", "kstTrkmPt", "kstTrkpPt", "bPhi", "kstPhi", "mumuPhi", "mumPhi",
        "mupPhi", "kstTrkmPhi", "kstTrkpPhi", "bEta", "kstEta", "mumuEta", "mumEta",
        "mupEta", "kstTrkmEta", "kstTrkpEta", "tagB0",


    };

	vector<string> var_plots = {
		"bMass", "bVtxCL", "mumuMass", "bPt", "kstPt", "mumuPt", "mumPt", "mupPt", "kstTrkmPt", 
		"kstMass", "kstTrkpPt", "bPhi", "kstPhi", "mumuPhi", "mumPhi", "mupPhi", "kstTrkmPhi", "kstTrkpPhi", 
		"bEta", "kstEta", "mumuEta", "mumEta", "mupEta", "kstTrkmEta", "kstTrkpEta"
	};

	// Allocate variables and set branch addresses
	for (const auto &name : variables){
		vars_data[name] = 0;
		vars_mc[name] = 0;
		t_data->SetBranchAddress(name.c_str(), &vars_data[name]);
		t_mc->SetBranchAddress(name.c_str(), &vars_mc[name]);
	}
	
	// Histogram storage
	map<string, TH1D*> h_data;
	map<string, TH1D*> h_mc;

	// Histogram parameters
	map<string, tuple<int, double, double>> histParams = {
		{"bMass", {100, 4.8, 5.8}}, {"bPt", {100, 0, 30}}, {"bEta", {100, -3, 3}}, {"bPhi", {100, -3.5, 3.5}}, {"bVtxCL", {100, 0, 1}},
		{"kstMass", {100, 0.5, 1.5}}, {"kstPt", {100, 0, 5}}, {"kstEta", {100, -3, 3}}, {"kstPhi", {100, -3.5, 3.5}},
		{"mumuMass", {100, 0, 4.5}}, {"mumuPt", {100, 0, 30}}, {"mumuEta", {100, -3, 3}}, {"mumuPhi", {100, -3.5, 3.5}},
		{"kstTrkmPt", {100, 0, 5}}, {"kstTrkmEta", {100, -3, 3}}, {"kstTrkmPhi", {100, -3.5, 3.5}},
		{"kstTrkpPt", {100, 0, 5}}, {"kstTrkpEta", {100, -3, 3}}, {"kstTrkpPhi", {100, -3.5, 3.5}},
		{"mumPt", {100, 0, 25}}, {"mumEta", {100, -3, 3}}, {"mumPhi", {100, -3.5, 3.5}},
		{"mupPt", {100, 0, 25}}, {"mupEta", {100, -3, 3}}, {"mupPhi", {100, -3.5, 3.5}},
	};

	map<string, string> axisTitles = {
        {"bMass", "m(B^{0}) [GeV]"}, {"bPt", "p_{T}(B^{0}) [GeV]"}, {"bEta", "#eta(B)"},
        {"bPhi", "#phi(B) [rad]"}, {"bVtxCL", "Vertex CL"},
        {"kstMass", "m(K*) [GeV]"}, {"kstPt", "p_{T}(K*) [GeV]"}, 
        {"kstEta", "#eta(K*)"}, {"kstPhi", "#phi(K*) [rad]"},
        {"mumuMass", "m(#mu#mu) [GeV]"}, {"mumuPt", "p_{T}(#mu#mu) [GeV]"},
        {"mumuEta", "#eta(#mu#mu)"}, {"mumuPhi", "#phi(#mu#mu) [rad]"},
        {"kstTrkmPt", "Negative track p_{T} [GeV]"}, {"kstTrkmEta", "Negative track #eta"}, 
        {"kstTrkmPhi", "Negative track #phi [rad]"}, {"kstTrkpPt", "Positive track p_{T} [GeV]"}, 
        {"kstTrkpEta", "Positive track #eta"}, {"kstTrkpPhi", "Positive track #phi [rad]"},
        {"mumPt", "p_{T}(#mu^{--}) [GeV]"}, {"mumEta", "#eta(#mu^{--})"}, {"mumPhi", "#phi(#mu^{--}) [rad]"},
        {"mupPt", "p_{T}(#mu^{+}) [GeV]"}, {"mupEta", "#eta(#mu^{+})"}, {"mupPhi", "#phi(#mu^{+}) [rad]"},   
	};

	// Create histograms 
	for (const auto &name : var_plots){
		auto [bins, xmin, xmax] = histParams[name];
		h_data[name] = new TH1D(("h_data_" + name).c_str(), "", bins, xmin, xmax);
        h_mc[name]   = new TH1D(("h_mc_"   + name).c_str(), "", bins, xmin, xmax);
		h_data[name]->GetXaxis()->SetTitle(axisTitles[name].c_str());
        h_data[name]->GetYaxis()->SetTitle("Events / Bin (Normalized)");
	}

	// Fill data histograms   
	for (int i = 0; i < t_data->GetEntries(); i++){
		t_data->GetEntry(i);

		double mass_b_data = (vars_data["tagB0"] == 1) ? vars_data["bMass"] : vars_data["bBarMass"];
		if (mass_b_data < 5.0 || mass_b_data > 5.6) continue;

		double mass_kst_data = (vars_data["tagB0"] == 1) ? vars_data["kstMass"] : vars_data["kstBarMass"];
		
		// fill histograms 
		for (const auto &name : var_plots){
			if (name == "bMass" || name == "kstMass") continue;
			h_data[name]->Fill(vars_data[name]);
		}

		h_data["bMass"]->Fill(mass_b_data);
        h_data["kstMass"]->Fill(mass_kst_data);
	}
		
	// Fill mc histograms
	for (int i = 0; i < t_mc->GetEntries(); i++){
		t_mc->GetEntry(i);

		double mass_b_mc = (vars_mc["tagB0"] == 1) ? vars_mc["bMass"] : vars_mc["bBarMass"];
		if (mass_b_mc < 5.0 || mass_b_mc > 5.6) continue;

		double mass_kst_mc = (vars_mc["tagB0"] == 1) ? vars_mc["kstMass"] : vars_mc["kstBarMass"];

		for (const auto &name : var_plots){
			if (name == "bMass" || name == "kstMass") continue;
			h_mc[name]->Fill(vars_mc[name]);
		}

		h_mc["bMass"]->Fill(mass_b_mc);
        h_mc["kstMass"]->Fill(mass_kst_mc);

	}

	// Normalise histograms
	for (const auto &name : var_plots){
		if (h_data[name]->Integral() > 0)
			h_data[name]->Scale(1.0 / h_data[name]->Integral());
		if (h_mc[name]->Integral() > 0)
			h_mc[name]->Scale(1.0 / h_mc[name]->Integral());
	}

	// Draw histograms
	for (const auto &name : var_plots){
		TCanvas *c = new TCanvas(("c_" + name).c_str(), name.c_str());

		double max_val = max(h_data[name]->GetMaximum(), h_mc[name]->GetMaximum());
		h_data[name]->SetMaximum(1.1 * max_val); 

        h_data[name]->SetLineColor(kBlue);
        h_data[name]->SetFillColorAlpha(kBlue, 0.5);

		h_mc[name]->SetLineColor(kRed);
        h_mc[name]->SetFillColorAlpha(kRed, 0.3);

		h_data[name]->Draw("HIST");
		h_mc[name]->Draw("HIST SAME");

        
        // Add legend  
        TLegend *leg = new TLegend(0.65, 0.75, 0.88, 0.88);
        leg->AddEntry(h_data[name], "Data", "f");
        leg->AddEntry(h_mc[name], "MC", "f");
		leg->SetTextSize(0.03);
        leg->SetBorderSize(0);
        leg->Draw();

        c->SaveAs(("DataMC_Plots/" + name + "_comparison.png").c_str());
	}

	for (const auto &name : var_plots){
        delete h_data[name];
        delete h_mc[name];
    }

	// Close files
	f_data->Close();    
	f_mc->Close();
	
}
