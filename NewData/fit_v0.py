"""
Module do fit B0 mass spectrum with RooFit
"""
import os
import ROOT
from ROOT import TFile
from ROOT import TLorentzVector
from ROOT import TH1F
from ROOT import RooRealVar
from ROOT import RooDataHist
from ROOT import RooDataSet
from ROOT import RooExponential
from ROOT import RooGaussian
from ROOT import RooCBShape
from ROOT import RooArgList
from ROOT import RooArgSet
from ROOT import RooAddPdf
from ROOT import RooPlot
from ROOT import TLegend
from ROOT import RooFit
from ROOT import TLatex

import json
import numpy as np

def fit_mc():
    """Fit MC with pure signal (double Gaussian). Save params for later use and plot."""

    mmin, mmax = 5.0, 5.6

    f = TFile.Open("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22Jpsi.root")
    h = TH1F("h", "", 100, mmin, mmax)

    #Filling the histogram with the relevant data
    tree = f.Get("ntuple")
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.tagB0 == 1:
            mass = tree.bMass
        elif tree.tagB0 == 0:
            mass = tree.bBarMass

        if (mass < mmin) or (mass > mmax): 
            continue
        
        # truth matching
        if (tree.truthMatchMum == 0 or tree.truthMatchMup == 0 or 
            tree.truthMatchTrkm == 0 or tree.truthMatchTrkp == 0):
            continue

        h.Fill(mass)

    # RooFit objects
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    args = RooArgList(mass)
    dh = RooDataHist("dh_mc", "dh_mc", args , h)

    # Signal model: Double gaussian
    mean = RooRealVar("mean", "mean", 0.5*(mmin + mmax), mmin, mmax)
    sigma1 = RooRealVar("sigma1", "sigma1", 0.01, 0., 0.2)   
    sigma2 = RooRealVar("sigma2", "sigma2", 0.03, 0., 0.2)   
    frac = RooRealVar("frac", "frac", 0.6, 0., 1.0)

    signal1 = RooGaussian("signal1", "signal1", mass, mean, sigma1)
    signal2 = RooGaussian("signal2", "signal2", mass, mean, sigma2)
    signal = RooAddPdf("signal", "signal", RooArgList(signal1, signal2), RooArgList(frac))

    # Background
    Lambda = RooRealVar("lambda", "lambda", -0.5, -10.0, 0.0)
    background = RooExponential("background", "background", mass, Lambda)

    # Yields
    nsig = RooRealVar("nsig", "nsig", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    nbkg = RooRealVar("nbkg", "nbkg", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

    # Fit
    model.fitTo(dh, RooFit.Extended(True))

    # Effective sigma
    sigma_eff = np.sqrt(frac.getVal()*sigma1.getVal()**2 + (1-frac.getVal())*sigma2.getVal()**2)

    # --- Plot ---
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh_mc"))
    model.plotOn(frame, RooFit.Name("model"))
    model.plotOn(frame, RooFit.Components("background"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("bkg"))
    model.plotOn(frame, RooFit.Components("signal"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen+2),
                 RooFit.Name("sig"))
    # individual Gaussians
    #signal1.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.LineStyle(ROOT.kDotted), RooFit.Name("sig1"))
    #signal2.plotOn(frame, RooFit.LineColor(ROOT.kMagenta), RooFit.LineStyle(ROOT.kDotted), RooFit.Name("sig2"))

    c = ROOT.TCanvas()
    frame.Draw()
    frame.SetTitle("")

    legend = TLegend(0.65, 0.6, 0.88, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextFont(40)
    legend.SetTextSize(0.03)
    legend.AddEntry(frame.findObject("dh_mc"), "MC", "lep")
    legend.AddEntry(frame.findObject("sig"), "Signal (double Gauss)", "l")
    legend.AddEntry(frame.findObject("bkg"), "Background", "l")
    legend.AddEntry(frame.findObject("model"), "Global fit", "l")
    #legend.AddEntry(frame.findObject("sig1"), "Gauss 1", "l")
    #legend.AddEntry(frame.findObject("sig2"), "Gauss 2", "l")
    legend.Draw()

    # Fit results
    L = TLatex()
    L.SetNDC()
    L.SetTextSize(0.02)
    y_start, step = 0.85, 0.035
    L.DrawLatex(0.15, y_start, ROOT.Form("Y_{S}: %.0f #pm %.0f" % (nsig.getVal(), nsig.getError())))
    L.DrawLatex(0.15, y_start - step, ROOT.Form("Y_{B}: %.0f #pm %.0f" % (nbkg.getVal(), nbkg.getError())))
    L.DrawLatex(0.15, y_start - 2*step, ROOT.Form("Mean: %6.4f #pm %6.4f GeV/c^{2}" % (mean.getVal(), mean.getError())))
    L.DrawLatex(0.15, y_start - 3*step, ROOT.Form("#sigma_{1}: %6.4f #pm %6.4f MeV/c^{2}" % (sigma1.getVal()*1000, sigma1.getError()*1000)))
    L.DrawLatex(0.15, y_start - 4*step, ROOT.Form("#sigma_{2}: %6.4f #pm %6.4f MeV/c^{2}" % (sigma2.getVal()*1000, sigma2.getError()*1000)))
    L.DrawLatex(0.15, y_start - 5*step, ROOT.Form("Frac: %5.3f #pm %5.3f" % (frac.getVal(), frac.getError())))
    L.DrawLatex(0.15, y_start - 6*step, ROOT.Form("#lambda: %6.4f #pm %6.4f" % (Lambda.getVal(), Lambda.getError())))
    L.DrawLatex(0.15, y_start - 7*step, ROOT.Form("#sigma_{eff}: %6.3f MeV/c^{2}" % (sigma_eff*1000)))

    # Compute chi2 manually 
    chi2 = 0.0
    for i in range(1, h.GetNbinsX() + 1):  # ROOT histogram bins start at 1
        observed = h.GetBinContent(i)
        error = h.GetBinError(i)
        if error == 0:  # avoid division by zero
            continue
        x_val = h.GetBinCenter(i)
        mass.setVal(x_val)
        expected = model.getVal(RooArgSet(mass)) * h.GetBinWidth(i) * dh.sumEntries()  # scale by bin width & entries
        chi2 += (observed - expected)**2 / (error**2)

    variables = 7 # free parameters
    ndf = h.GetNbinsX() - variables
    chi2_ndf = chi2 / ndf
    L.DrawLatex(0.15, y_start - 8*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))

    # Save MC parameters
    params = {
        "mean": (mean.getVal(), mean.getError()),
        "sigma1": (sigma1.getVal(), sigma1.getError()),
        "sigma2": (sigma2.getVal(), sigma2.getError()),
        "frac": (frac.getVal(), frac.getError()),
        "lambda": (Lambda.getVal(), Lambda.getError()),
        "nsig": (nsig.getVal(), nsig.getError()),
        "nbkg": (nbkg.getVal(), nbkg.getError()),
        "sigma_eff": sigma_eff,
        "chi2_ndf": chi2_ndf
    }

    # Sidebands
    sb_left_min, sb_left_max = mmin, mean.getVal() - 3*sigma_eff
    sb_right_min, sb_right_max = mean.getVal() + 3*sigma_eff, mmax
    params.update({
        "sb_left_min": sb_left_min, "sb_left_max": sb_left_max,
        "sb_right_min": sb_right_min, "sb_right_max": sb_right_max
    })

    os.makedirs("NewData/scalings", exist_ok=True)
    with open("NewData/scalings/fit_params_mc.json", "w") as fout:
        json.dump(params, fout, indent=4)

    c.Draw()
    c.SaveAs("NewData/bMassPlots/fit_mc.pdf")
    f.Close()


def fit_data():
    """Fit data with exponential + fixed signal from MC. Save params and plot."""
    mmin, mmax = 5.0, 5.6

    f = TFile.Open("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22F.root")
    h = TH1F("h_data", "", 100, mmin, mmax)

    #Filling the histogram with the relevant data
    tree = f.Get("ntuple")
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.tagB0 == 1:
            mass = tree.bMass
        elif tree.tagB0 == 0:
            mass = tree.bBarMass

        if (mass < mmin) or (mass > mmax): 
            continue
        
        # selection cuts 
        if tree.bVtxCL < 0.2: continue
        if tree.bCosAlphaBS < 0.9955: continue
        if tree.bLBS < 0.05: continue

        h.Fill(mass)

    # RooFit dataset
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh = RooDataHist("dh_data", "dh_data", RooArgList(mass), h)

    # Load MC params
    with open("NewData/scalings/fit_params_mc.json") as fmc:
        mc_params = json.load(fmc)

    # Fixed signal
    mean = RooRealVar("mean", "mean", mc_params["mean"][0]); mean.setConstant(True)
    sigma1 = RooRealVar("sigma1", "sigma1", mc_params["sigma1"][0]); sigma1.setConstant(True)
    sigma2 = RooRealVar("sigma2", "sigma2", mc_params["sigma2"][0]); sigma2.setConstant(True)
    frac = RooRealVar("frac", "frac", mc_params["frac"][0]); frac.setConstant(True)

    signal1 = RooGaussian("signal1", "signal1", mass, mean, sigma1)
    signal2 = RooGaussian("signal2", "signal2", mass, mean, sigma2)
    signal = RooAddPdf("signal", "signal", RooArgList(signal1, signal2), RooArgList(frac))

    # Background
    Lambda = RooRealVar("lambda", "lambda", -1.0, -10.0, 0.0)
    background = RooExponential("background", "background", mass, Lambda)

    # Yields
    nsig = RooRealVar("nsig", "nsig", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    nbkg = RooRealVar("nbkg", "nbkg", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

    # Fit
    model.fitTo(dh, RooFit.Extended(True))

    # Compute background yields in regions
    sb_left_min, sb_left_max = mc_params["sb_left_min"], mc_params["sb_left_max"]
    sb_right_min, sb_right_max = mc_params["sb_right_min"], mc_params["sb_right_max"]
    
    mass.setRange("SR", sb_left_max, sb_right_min)
    mass.setRange("SB_left", sb_left_min, sb_left_max)
    mass.setRange("SB_right", sb_right_min, sb_right_max)

    # get normalized fractions (fraction of background PDF inside each region)
    frac_SR = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SR")).getVal()
    frac_SB_left = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_left")).getVal()
    frac_SB_right = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_right")).getVal()

    # expected background event counts in each region
    nbkg_SR = nbkg.getVal() * frac_SR
    nbkg_SB = nbkg.getVal() * (frac_SB_left + frac_SB_right)

    # --- Plot ---
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh_data"))
    model.plotOn(frame, RooFit.Name("model"))
    model.plotOn(frame, RooFit.Components("background"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("bkg"))
    model.plotOn(frame, RooFit.Components("signal"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen+2),
                 RooFit.Name("sig"))

    c = ROOT.TCanvas()
    frame.Draw()
    frame.SetTitle("")

    legend = TLegend(0.65, 0.6, 0.88, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextFont(40)
    legend.SetTextSize(0.03)
    legend.AddEntry(frame.findObject("dh_data"), "Data", "lep")
    legend.AddEntry(frame.findObject("sig"), "Signal (MC shape)", "l")
    legend.AddEntry(frame.findObject("bkg"), "Background (exp)", "l")
    legend.AddEntry(frame.findObject("model"), "Global fit", "l")
    legend.Draw()

    L = TLatex()
    L.SetNDC()
    L.SetTextSize(0.02)
    y_start, step = 0.85, 0.035
    L.DrawLatex(0.15,y_start, ROOT.Form("Y_{S}: %.0f #pm %.0f events" % (nsig.getVal(),nsig.getError())))
    L.DrawLatex(0.15,y_start - step, ROOT.Form("Y_{B}: %.0f #pm %.0f events" % (nbkg.getVal(),nbkg.getError())))
    L.DrawLatex(0.15,y_start - 2*step, ROOT.Form("Mean: %5.4f #pm %5.4f GeV/c^{2}" % (mean.getVal(), mc_params["mean"][1])))
    L.DrawLatex(0.15,y_start - 3*step, ROOT.Form("#lambda: %5.4f #pm %5.4f GeV^{-1}" % (Lambda.getVal(), Lambda.getError())))
    L.DrawLatex(0.15, y_start - 4*step, ROOT.Form("#sigma_{1fixed}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma1.getVal()*1000, mc_params["sigma1"][1]*1000)))
    L.DrawLatex(0.15, y_start - 5*step, ROOT.Form("#sigma_{2fixed}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma2.getVal()*1000, mc_params["sigma2"][1]*1000)))

    # Compute chi2 manually
    nbins = dh.numEntries()  # total number of bins
    chi2 = 0.0

    for i in range(1, h.GetNbinsX() + 1):  # ROOT histogram bins start at 1
        observed = h.GetBinContent(i)
        error = h.GetBinError(i)
        if error == 0:  # avoid division by zero
            continue
        x_val = h.GetBinCenter(i)
        mass.setVal(x_val)
        expected = model.getVal(RooArgSet(mass)) * h.GetBinWidth(i) * dh.sumEntries()  # scale by bin width & entries
        chi2 += (observed - expected)**2 / (error**2)

    variables = 3  # free parameters
    ndf = h.GetNbinsX() - variables
    chi2_ndf = chi2 / ndf

    # Draw on plot
    L.DrawLatex(0.15, y_start - 6*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))

    c.Draw()
    c.SaveAs("NewData/bMassPlots/fit_data.pdf")
    f.Close()

    # Save results
    params = {
        "nsig": (nsig.getVal(), nsig.getError()),
        "nbkg": (nbkg.getVal(), nbkg.getError()),
        "lambda": (Lambda.getVal(), Lambda.getError()),
        "nbkg_SR": nbkg_SR,
        "nbkg_SB": nbkg_SB,
        "chi2_ndf": chi2_ndf
    }
    with open("NewData/scalings/fit_params_data.json", "w") as fout:
        json.dump(params, fout, indent=4)



if __name__ == "__main__":

    os.makedirs("NewData/bMassPlots", exist_ok=True)

    fit_mc()
    fit_data()