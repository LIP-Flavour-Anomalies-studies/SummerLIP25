#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;

void bMass(){

    gStyle->SetOptStat(0); 

    // --- Open Files and Get Trees ---
    TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	
	TTree *t_data = (TTree*)f_data->Get("ntuple");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");


    // --- Declare variables for data ---
    double data_bMass, data_bBarMass, data_tagB0;
    double data_bVtxCL;
    double data_mumuMass;
    
    t_data->SetBranchAddress("bMass", &data_bMass);
    t_data->SetBranchAddress("bBarMass", &data_bBarMass);
    t_data->SetBranchAddress("tagB0", &data_tagB0);
    t_data->SetBranchAddress("bVtxCL", &data_bVtxCL);
    t_data->SetBranchAddress("mumuMass", &data_mumuMass);

    // --- Declare variables for simulation ---
    double mc_bMass, mc_bBarMass, mc_tagB0;
    double mc_bVtxCL;
    double mc_mumuMass;
    double mc_truthMatchMum, mc_truthMatchMup, mc_truthMatchTrkm, mc_truthMatchTrkp;

    t_mc->SetBranchAddress("bMass", &mc_bMass);
    t_mc->SetBranchAddress("bBarMass", &mc_bBarMass);
    t_mc->SetBranchAddress("tagB0", &mc_tagB0);
    t_mc->SetBranchAddress("bVtxCL", &mc_bVtxCL);
    t_mc->SetBranchAddress("mumuMass", &mc_mumuMass);

    t_mc->SetBranchAddress("truthMatchMum", &mc_truthMatchMum);
    t_mc->SetBranchAddress("truthMatchMup", &mc_truthMatchMup);
    t_mc->SetBranchAddress("truthMatchTrkm", &mc_truthMatchTrkm);
    t_mc->SetBranchAddress("truthMatchTrkp", &mc_truthMatchTrkp);

    // --- Create histograms ---
    TH1D *h_bMass_data = new TH1D("h_bMass_data", "B0 Mass (Data)", 100, 4.8, 5.8);
    TH1D *h_bMass_mc_truth_matched = new TH1D("h_bMass_mc_truth_matched", "B0 Mass (MC Truth Matched)", 100, 4.8, 5.8);

    h_bMass_data->GetXaxis()->SetTitle("m(B^{0}) [GeV/c^{2}]");
    h_bMass_data->GetYaxis()->SetTitle("Events / Bin (Normalized)");
    h_bMass_mc_truth_matched->GetXaxis()->SetTitle("m(B^{0}) [GeV/c^{2}]");
    h_bMass_mc_truth_matched->GetYaxis()->SetTitle("Events / Bin (Normalized)");

    // --- Fill data histogram ---
    cout << "Processing Data Events..." << endl;
    Long64_t nEntries_data = t_data->GetEntries();
    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);

        //Selection cuts
        if (data_bVtxCL < 0.01) continue; // skip events with poor vertex fit   

        double bMass_data = (data_tagB0 == 1) ? data_bMass : data_bBarMass;
		if (bMass_data < 5.0 || bMass_data > 5.6) continue;

        h_bMass_data->Fill(bMass_data);
    }
    cout << "Finished processing " << nEntries_data << " Data Events." << endl;

    // --- Fill MC histogram ---
    cout << "Processing MC Events..." << endl;
    Long64_t nEntries_mc = t_mc->GetEntries();
    int matchedMC = 0;
    for(Long64_t i = 0; i < nEntries_mc; i++){    
        t_mc->GetEntry(i);

        // Truth matching cuts first to isolate signal
        if (mc_truthMatchMum == 0) continue;
        if (mc_truthMatchMup == 0) continue;
        if (mc_truthMatchTrkm == 0) continue;
        if (mc_truthMatchTrkp == 0) continue;

        matchedMC++;

        //Selection cuts
        //if (mc_bVtxCL < 0.01) continue; // skip events with poor vertex fit   

        double bMass_mc = (mc_tagB0 == 1) ? mc_bMass : mc_bBarMass;
		if (bMass_mc < 5.134 || bMass_mc > 5.416) continue;

        h_bMass_mc_truth_matched->Fill(bMass_mc);
    }
    cout << "Total MC events: " << nEntries_mc << endl;
    cout << "Fully truth-matched MC events: " << matchedMC << endl;

    // --- Normalise histograms ---
    // For initial shape comparison, normalize to integral.
    // For a final result, would scale MC by luminosity and cross-section.
    if (h_bMass_data->Integral() > 0)
        h_bMass_data->Scale(1.0 / h_bMass_data->Integral());
    if (h_bMass_mc_truth_matched->Integral() > 0)
        h_bMass_mc_truth_matched->Scale(1.0 / h_bMass_mc_truth_matched->Integral());

    //--- Draw and save comparison plot ---
    TCanvas *c_bMass = new TCanvas("c_bMass", "B0 Mass Comparison", 800, 600);
    c_bMass->SetLeftMargin(0.12);
    c_bMass->SetBottomMargin(0.12);


    double max_val = max(h_bMass_data->GetMaximum(), h_bMass_mc_truth_matched->GetMaximum());
    h_bMass_data->SetMaximum(max_val * 1.2); // Add some padding above the max
    
    // Data style
    h_bMass_data->SetLineColor(kBlue);
    h_bMass_data->SetLineWidth(2);

    // MC style
    h_bMass_mc_truth_matched->SetLineColor(kRed);
    h_bMass_mc_truth_matched->SetLineWidth(2);

    // Draw
    h_bMass_data->Draw("HIST"); // Draw data with error bars and points
    h_bMass_mc_truth_matched->Draw("HIST SAME"); // Draw MC as filled histogram

    // Create Legend
    TLegend *leg = new TLegend(0.70, 0.75, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.035);
    leg->AddEntry(h_bMass_data, "Data", "lp");
    leg->AddEntry(h_bMass_mc_truth_matched, "MC", "lf");
    leg->Draw();

    c_bMass->SaveAs("Data_vs_MC_TruthMatched.png");

    f_data->Close();
    f_mc->Close();
}