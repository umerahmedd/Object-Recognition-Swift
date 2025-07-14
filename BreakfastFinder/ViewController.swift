/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Contains the view controller for the Breakfast Finder.
*/

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var bufferSize: CGSize = .zero
    var rootLayer: CALayer! = nil
    
    @IBOutlet weak private var previewView: UIView!
    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private let videoDataOutput = AVCaptureVideoDataOutput()
    
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // to be implemented in the subclass
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupAVCapture()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func setupAVCapture() {
        var deviceInput: AVCaptureDeviceInput!
        
        // Select a video device, make an input
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        // Use highest resolution preset for better recognition accuracy
        session.sessionPreset = .hd1920x1080
        
        // Add a video input
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        let captureConnection = videoDataOutput.connection(with: .video)
        // Always process the frames
        captureConnection?.isEnabled = true
        
        // Remove the fixed orientation setting to allow dynamic orientation
        // The orientation will be handled by the EXIF orientation in Vision processing
        
        do {
            try videoDevice!.lockForConfiguration()
            
            // Configure for highest frame rate and optimal performance
            if let format = selectOptimalFormat(for: videoDevice!) {
                videoDevice!.activeFormat = format
                
                // Set maximum frame rate for smoother recognition
                let maxFrameRate = format.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 30
                videoDevice!.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(maxFrameRate))
                videoDevice!.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(maxFrameRate))
            }
            
            // Enable smooth auto focus for consistent recognition
            if videoDevice!.isSmoothAutoFocusSupported {
                videoDevice!.isSmoothAutoFocusEnabled = true
            }
            
            // Set focus mode for optimal object detection
            if videoDevice!.isFocusModeSupported(.continuousAutoFocus) {
                videoDevice!.focusMode = .continuousAutoFocus
            }
            
            // Enable auto exposure for consistent lighting
            if videoDevice!.isExposureModeSupported(.continuousAutoExposure) {
                videoDevice!.exposureMode = .continuousAutoExposure
            }
            
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch {
            print(error)
        }
        session.commitConfiguration()
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        rootLayer = previewView.layer
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
    }
    
    // Helper function to select the optimal camera format
    private func selectOptimalFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let formats = device.formats
        var bestFormat: AVCaptureDevice.Format?
        var bestScore = 0
        
        for format in formats {
            let description = format.formatDescription
            let dimensions = CMVideoFormatDescriptionGetDimensions(description)
            let frameRateRanges = format.videoSupportedFrameRateRanges
            
            // Prefer higher resolution (1080p or better) with high frame rate
            let resolution = Int(dimensions.width * dimensions.height)
            let maxFrameRate = frameRateRanges.first?.maxFrameRate ?? 0
            
            // Score based on resolution and frame rate
            let score = resolution + Int(maxFrameRate * 10000)
            
            if score > bestScore {
                bestScore = score
                bestFormat = format
            }
        }
        
        return bestFormat
    }
    
    func startCaptureSession() {
        session.startRunning()
    }
    
    // Clean up capture setup
    func teardownAVCapture() {
        previewLayer.removeFromSuperlayer()
        previewLayer = nil
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput, didDrop didDropSampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // print("frame dropped")
    }
    
    public func exifOrientationFromDeviceOrientation() -> CGImagePropertyOrientation {
        let interfaceOrientation = UIApplication.shared.statusBarOrientation
        let exifOrientation: CGImagePropertyOrientation
        
        switch interfaceOrientation {
        case .portraitUpsideDown:  // Device oriented vertically, home button on the top
            exifOrientation = .left
        case .landscapeLeft:       // Device oriented horizontally, home button on the right
            exifOrientation = .upMirrored
        case .landscapeRight:      // Device oriented horizontally, home button on the left
            exifOrientation = .down
        case .portrait:            // Device oriented vertically, home button on the bottom
            exifOrientation = .up
        default:
            exifOrientation = .up
        }
        return exifOrientation
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        if previewLayer != nil {
            previewLayer.frame = rootLayer.bounds
            updatePreviewLayerOrientation()
        }
    }
    
    func updatePreviewLayerOrientation() {
        guard let previewLayer = previewLayer else { return }
        
        let orientation = UIApplication.shared.statusBarOrientation
        var videoOrientation: AVCaptureVideoOrientation = .portrait
        
        switch orientation {
        case .portrait:
            videoOrientation = .portrait
        case .portraitUpsideDown:
            videoOrientation = .portraitUpsideDown
        case .landscapeLeft:
            videoOrientation = .landscapeLeft
        case .landscapeRight:
            videoOrientation = .landscapeRight
        default:
            videoOrientation = .portrait
        }
        
        if previewLayer.connection?.isVideoOrientationSupported == true {
            previewLayer.connection?.videoOrientation = videoOrientation
        }
    }
}

