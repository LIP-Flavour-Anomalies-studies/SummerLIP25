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

void comparison(){

    gStyle->SetOptStat(0);

    // --- Open Files and Get Trees ---
    TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TTree *t_data = (TTree*)f_data->Get("ntuple");

    TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");

    // --- Output files and trees ---
    TFile *f_comparison = new TFile("comparison.root", "RECREATE");
    TTree *t_background = new TTree("Tback", "Background Sidebands");
    TTree *t_signal = new TTree("Tsignal", "MC Signal");

    // --- Storage for input/output variable values ---
	map<string, double> vars_data;
	map<string, double> vars_mc;
    map<string, double> vars_signal;
    map<string, double> vars_background;

    // --- Variable names ---
    // common varaibles 
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

    // variables to plot 
	vector<string> var_plots = {
        "bMass", "bVtxCL", "bPt", "bPhi", "bEta",
        "kstMass", "kstPt", "kstPhi", "kstEta",
        "mumuMass", "mumuPt", "mumuPhi", "mumuEta",
        "kstTrkmPt", "kstTrkmPhi", "kstTrkmEta", "kstTrkmDCABS",
        "kstTrkpPt", "kstTrkpPhi",  "kstTrkpEta", "kstTrkpDCABS",
        "mumPt", "mumPhi", "mumEta", 
        "mupPt", "mupPhi", "mupEta",
        "bCosAlphaBS", "bLBS", "bDCABS",
        "muLeadingPt", "muTrailingPt"
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
    for (const auto &name : var_plots){
        vars_signal[name] = 0;
        vars_background[name] = 0;
        t_signal->Branch(name.c_str(), &vars_signal[name]);
        t_background->Branch(name.c_str(), &vars_background[name]);
    }

    double mmin = 5;
    double mmax = 5.6;
    double s_left = 5.15;
    double s_right = 5.4;
    double nbins = 100;

    // --- Histogram storage ---
    map<string, TH1D*> h_data;
	map<string, TH1D*> h_mc;

    // --- Histogram Parameters ---
    map<string, tuple<int, double, double>> histParams = {
		{"bMass", {nbins, mmin, mmax}}, {"bPt", {nbins, 0, 40}}, {"bEta", {nbins, -3, 3}}, {"bPhi", {nbins, -3.5, 3.5}}, {"bVtxCL", {nbins, 0, 1}},
		{"kstMass", {nbins, 0.5, 1.5}}, {"kstPt", {nbins, 0, 5}}, {"kstEta", {nbins, -3, 3}}, {"kstPhi", {nbins, -3.5, 3.5}},
		{"mumuMass", {nbins, 0, 4.5}}, {"mumuPt", {nbins, 0, 30}}, {"mumuEta", {nbins, -3, 3}}, {"mumuPhi", {nbins, -3.5, 3.5}},
		{"kstTrkmPt", {nbins, 0, 5}}, {"kstTrkmEta", {nbins, -3, 3}}, {"kstTrkmPhi", {nbins, -3.5, 3.5}},
		{"kstTrkpPt", {nbins, 0, 5}}, {"kstTrkpEta", {nbins, -3, 3}}, {"kstTrkpPhi", {nbins, -3.5, 3.5}},
		{"mumPt", {nbins, 0, 30}}, {"mumEta", {nbins, -3, 3}}, {"mumPhi", {nbins, -3.5, 3.5}},
		{"mupPt", {nbins, 0, 30}}, {"mupEta", {nbins, -3, 3}}, {"mupPhi", {nbins, -3.5, 3.5}},
        {"bCosAlphaBS", {nbins, 0.8, 1.}}, {"bLBS", {nbins, 0., 1.}}, {"bDCABS", {nbins, -0.05, 0.05}},
        {"kstTrkmDCABS", {nbins, -1, 1}}, {"kstTrkpDCABS", {nbins, -1, 1}},
        {"muLeadingPt", {nbins, 0, 30}}, {"muTrailingPt", {nbins, 0, 30}}
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
        {"bCosAlphaBS", "cos(#alpha_{BS})"}, {"bLBS", "Flight length [cm]"}, {"bDCABS", "B^{0} DCA from BS [cm]"}, 
        {"kstTrkmDCABS", "Negative track DCA from BS [cm]"}, {"kstTrkpDCABS", "Positive track DCA from BS [cm]"},
        {"muLeadingPt", "Leading muon p_{T} [GeV]"}, {"muTrailingPt", "Trailing muon p_{T} [GeV]"}   
    };

    // --- Create histograms ---
    for (const auto &name : var_plots){
        auto [bins, xmin, xmax] = histParams[name];
		h_data[name] = new TH1D(("h_data_" + name).c_str(), "", bins, xmin, xmax);
        h_data[name]->SetDirectory(0);
        h_mc[name]   = new TH1D(("h_mc_"   + name).c_str(), "", bins, xmin, xmax);
        h_mc[name]->SetDirectory(0);
        h_data[name]->GetXaxis()->SetTitle(axisTitles[name].c_str());
        h_data[name]->GetYaxis()->SetTitle("Events / Bin (Normalized)");
    }

    // --- Fill data Histograms ---
    cout << "Looping over data..." << endl;
    Long64_t nEntries_data = t_data->GetEntries();
    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);    

        double mass_b = (vars_data["tagB0"] == 1) ? vars_data["bMass"] : vars_data["bBarMass"];
		if (mass_b < mmin || mass_b > mmax) continue;

        double mass_kst = (vars_data["tagB0"] == 1) ? vars_data["kstMass"] : vars_data["kstBarMass"];

        double muLeadingPt = max(vars_data["mupPt"], vars_data["mumPt"]);
        double muTrailingPt = min(vars_data["mupPt"], vars_data["mumPt"]);

        if (mass_b < s_left || mass_b > s_right){
            for (const auto &name : var_plots){
                if (name == "bMass" || name == "kstMass") continue;
                if (name == "muLeadingPt" || name == "muTrailingPt") continue;
                vars_background[name] = vars_data[name];
                h_data[name]->Fill(vars_data[name]);
            }
            h_data["bMass"]->Fill(mass_b);
            h_data["kstMass"]->Fill(mass_kst);
            h_data["muTrailingPt"]->Fill(muTrailingPt);
            h_data["muLeadingPt"]->Fill(muLeadingPt);

            // Fill tree only after setting all variables correctly
            vars_background["bMass"] = mass_b;
            vars_background["kstMass"] = mass_kst;
            vars_background["muTrailingPt"] = muTrailingPt;
            vars_background["muLeadingPt"] = muLeadingPt;
            t_background->Fill();
        }  
    }
    cout << "Finished processing data" << endl;


    // --- Fill MC Histograms --- 
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

        if (mass_b > s_left && mass_b < s_right){
            for (const auto &name : var_plots){
                if (name == "bMass" || name == "kstMass") continue;
                if (name == "muLeadingPt" || name == "muTrailingPt") continue;
                vars_signal[name] = vars_mc[name];
                h_mc[name]->Fill(vars_mc[name]);
            }
            h_mc["bMass"]->Fill(mass_b);
            h_mc["kstMass"]->Fill(mass_kst);
            h_mc["muTrailingPt"]->Fill(muTrailingPt);
            h_mc["muLeadingPt"]->Fill(muLeadingPt);

            vars_signal["bMass"] = mass_b;
            vars_signal["kstMass"] = mass_kst;
            vars_signal["muTrailingPt"] = muTrailingPt;
            vars_signal["muLeadingPt"] = muLeadingPt;
            t_signal->Fill();
        }  
    }
    cout << "Finished processing MC" << endl;

    // --- Normalise Histograms ---
    cout << "Normalising histograms..." << endl;
    for (const auto &name : var_plots){
		if (h_data[name]->Integral() > 0)
			h_data[name]->Scale(1.0 / h_data[name]->Integral());
		if (h_mc[name]->Integral() > 0)
			h_mc[name]->Scale(1.0 / h_mc[name]->Integral());
	}
    
    // --- Draw Histograms ---
    cout << "Drawing histograms..." << endl;
    for (const auto &name : var_plots){

        TCanvas *c = new TCanvas(("c_" + name).c_str(), name.c_str());

        // Set log scale only for these variables (just to try out)
        vector<string> logYVars = {"bCosAlphaBS", "kstTrkmDCABS", "kstTrkpDCABS"};

        if (find(logYVars.begin(), logYVars.end(), name) != logYVars.end()) {
            c->SetLogy();
        }

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
        leg->AddEntry(h_data[name], "Background", "f");
        leg->AddEntry(h_mc[name], "Signal (MC)", "f");
		leg->SetTextSize(0.03);
        leg->SetBorderSize(0);
        leg->Draw();

        c->SaveAs(("PLOTS comparison/" + name + ".png").c_str());
    }

    // Clean up
    for (const auto &name : var_plots){
        delete h_data[name];
        delete h_mc[name];
    }

    f_data->Close();    
	f_mc->Close();
    f_comparison->cd();
    t_signal->Write();
    t_background->Write();
    f_comparison->Close();

}
