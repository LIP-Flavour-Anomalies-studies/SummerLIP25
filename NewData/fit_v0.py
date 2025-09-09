"""
Module do fit B0 mass spectrum with RooFit
"""
import ROOT
from ROOT import TFile
from ROOT import TLorentzVector
from ROOT import TH1F
import numpy as np
from ROOT import RooRealVar
from ROOT import RooDataHist
from ROOT import RooDataSet
from ROOT import RooExponential
from ROOT import RooGaussian
from ROOT import RooArgList
from ROOT import RooArgSet
from ROOT import RooAddPdf
from ROOT import RooPlot
from ROOT import TLegend
from ROOT import RooFit
from ROOT import TLatex
import json



def fit(data_type):

    mmin = 5.0
    mmax = 5.6

    if data_type == "data":
        f = TFile.Open("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22F.root")
    if data_type == "mc":
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

        if data_type == "data":
            # apply selection cuts
            if tree.bVtxCL < 0.2: continue
            if tree.bCosAlphaBS < 0.9955: continue
            if tree.bLBS < 0.05: continue

        if data_type == "mc":
            # do matching
            if (tree.truthMatchMum == 0 or tree.truthMatchMup == 0 or 
                tree.truthMatchTrkm == 0 or tree.truthMatchTrkp == 0):
                continue

        h.Fill(mass)

    #Create a Mass variable that RooFit can use, and importing the relevant dataset
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    args = RooArgList(mass)
    dh = RooDataHist("dh", "dh", args, h)

    #Define background model (exponential) and its parameters
    Lambda = RooRealVar("lambda", "lambda", -1.0, -10.0,  0.0)
    background = RooExponential("background", "background", mass, Lambda)

    #Define signal model (double Gaussian) and its parameters
    mean = RooRealVar("mean", "mean", 0.5*(mmin+mmax), mmin, mmax)
    sigma1 = RooRealVar("sigma1", "sigma1", 0.1*(mmax-mmin),0.,0.5*(mmax-mmin))
    sigma2 = RooRealVar("sigma2", "sigma2", 0.3*(mmax-mmin),0.,0.5*(mmax-mmin))
    signal1 = RooGaussian("signal1", "signal1", mass, mean, sigma1)
    signal2 = RooGaussian("signal2", "signal2", mass, mean, sigma2)
    frac = RooRealVar("frac", "frac", 0.5,  0., 1.) # fraction between the two gaussians
    signal = RooAddPdf("signal", "signal", RooArgList(signal1, signal2), RooArgList(frac))

    #Define variables for number of signal and background events
    nsig_initial = 0.8*dh.sumEntries()
    nbkg_initial = 0.2*dh.sumEntries()
    nsig = RooRealVar("nsig", "nsig", nsig_initial, 0., dh.sumEntries())
    nbkg = RooRealVar("nbkg","nbkg",nbkg_initial, 0., dh.sumEntries())

    #Sum signal and background models
    model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

    #Perform the fit
    model.fitTo(dh, RooFit.Extended(True))

    #Plot the fit
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh"))
    model.plotOn(frame,RooFit.Name("modelSig"),RooFit.Components("signal"),RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    model.plotOn(frame,RooFit.Name("modelBkg"),RooFit.Components("background"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame,RooFit.Name("model"))

    c = ROOT.TCanvas()
    frame.Draw()
    frame.SetTitle("")

    #Draw a caption
    legend = TLegend(0.65,0.6,0.88,0.85)
    legend.SetBorderSize(0)
    legend.SetTextFont(40)
    legend.SetTextSize(0.04)
    legend.AddEntry(frame.findObject("dh"),"Data","1pe")
    legend.AddEntry(frame.findObject("modelBkg"),"Background fit","1pe")
    legend.AddEntry(frame.findObject("modelSig"),"Signal fit","1pe")
    legend.AddEntry(frame.findObject("model"),"Global fit","1pe")
    legend.Draw()

    #Display info and fit results
    L = TLatex()
    L.SetNDC()
    L.SetTextSize(0.02)
    y_start = 0.85
    step = 0.035
    L.DrawLatex(0.15,y_start, ROOT.Form("Y_{S}: %.0f #pm %.0f events" % (nsig.getVal(),nsig.getError())))
    L.DrawLatex(0.15,y_start - step, ROOT.Form("Y_{B}: %.0f #pm %.0f events" % (nbkg.getVal(),nbkg.getError())))
    L.DrawLatex(0.15,y_start - 2*step, ROOT.Form("Mean: %5.4f #pm %5.4f GeV/c^{2}" % (mean.getVal(),mean.getError())))
    L.DrawLatex(0.15,y_start - 3*step, ROOT.Form("#lambda: %5.4f #pm %5.4f GeV^{-1}" % (Lambda.getVal(), Lambda.getError())))
    L.DrawLatex(0.15,y_start - 4*step,ROOT.Form("#sigma_{1}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma1.getVal()*1000, sigma1.getError()*1000)))
    L.DrawLatex(0.15,y_start - 5*step, ROOT.Form("#sigma_{2}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma2.getVal()*1000, sigma2.getError()*1000)))
    
    # Effective sigma
    sigma_eff = np.sqrt(frac.getVal()*sigma1.getVal()**2 + (1-frac.getVal())*sigma2.getVal()**2)
    L.DrawLatex(0.15,y_start - 6*step, ROOT.Form("#sigma_{eff}: %5.4f MeV/c^{2}" % (sigma_eff*1000)))

    # --- Define SR/SB depending on data_type ---
    if data_type == "mc":
        sb_left_min, sb_left_max = mmin, mean.getVal() - 3*sigma_eff
        sb_right_min, sb_right_max = mean.getVal() + 3*sigma_eff, mmax

    elif data_type == "data":
        # Load MC-defined regions
        with open("NewData/scalings/fit_params_mc.json") as fmc:
            mc_params = json.load(fmc)
            sb_left_min, sb_left_max = mc_params["sb_left_min"], mc_params["sb_left_max"]
            sb_right_min, sb_right_max = mc_params["sb_right_min"], mc_params["sb_right_max"]

    # Compute background yields in regions
    mass.setRange("SR", sb_left_max, sb_right_min)
    mass.setRange("SB_left", sb_left_min, sb_left_max)
    mass.setRange("SB_right", sb_right_min, sb_right_max)

    # get normalized fractions (fraction of background PDF inside each region)
    frac_SR = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SR")).getVal()
    frac_SB_left = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_left")).getVal()
    frac_SB_right = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_high")).getVal()

    # expected background event counts in each region
    nbkg_SR = nbkg.getVal() * frac_SR
    nbkg_SB = nbkg.getVal() * (frac_SB_left + frac_SB_right)

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

    variables = 7  # free parameters
    ndf = h.GetNbinsX() - variables
    chi2_ndf = chi2 / ndf

    # Draw on plot
    L.DrawLatex(0.15, y_start - 7*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))
    
    c.Draw()
    c.SaveAs(f"NewData/bMassPlots/fit_{data_type}.pdf")
    f.Close()

    # Save parameters
    params = {
        "nsig": (nsig.getVal(), nsig.getError()),
        "nbkg": (nbkg.getVal(), nbkg.getError()),
        "mean": (mean.getVal(), mean.getError()),
        "sigma1": (sigma1.getVal(), sigma1.getError()),
        "sigma2": (sigma2.getVal(), sigma2.getError()),
        "frac": (frac.getVal(), frac.getError()),
        "lambda": (Lambda.getVal(), Lambda.getError()),
        "sigma_eff": sigma_eff,
        "nbkg_SR": nbkg_SR,
        "nbkg_SB": nbkg_SB,
        "chi2_ndf": chi2_ndf
    }
    
    if data_type == "mc":
        params.update({
            "sb_left_min": sb_left_min, "sb_left_max": sb_left_max,
            "sb_right_min": sb_right_min, "sb_right_max": sb_right_max
        })

    # Save parameters to JSON
    with open(f"NewData/scalings/fit_params_{data_type}.json", "w") as fout:
        json.dump(params, fout, indent=4)



if __name__ == "__main__":
    params_dir = f"NewData/scalings"
    os.makedirs(params, exist_ok=True)

    # Always fit MC first to define sidebands
    fit("mc")
    fit("data")