/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Contains the object recognition view controller for the Breakfast Finder.
*/

import UIKit
import AVFoundation
import Vision

class VisionObjectRecognitionViewController: ViewController {
    
    private var detectionOverlay: CALayer! = nil
    private var detectionCountLabel: CATextLayer! = nil
    
    // Vision parts
    private var requests = [VNRequest]()
    
    // Detection smoothing
    private var previousDetections: [String: CGRect] = [:]
    private let smoothingFactor: CGFloat = 0.7
    
    // Colors for different detection boxes
    private let boxColors: [CGColor] = [
        CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0), // Red
        CGColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0), // Green
        CGColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 1.0), // Blue
        CGColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0), // Yellow
        CGColor(red: 1.0, green: 0.0, blue: 1.0, alpha: 1.0), // Magenta
        CGColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 1.0), // Cyan
        CGColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0), // Orange
        CGColor(red: 0.5, green: 0.0, blue: 1.0, alpha: 1.0)  // Purple
    ]
    
    @discardableResult
    func setupVision() -> NSError? {
        // Setup Vision parts
        let error: NSError! = nil
        
        guard let modelURL = Bundle.main.url(forResource: "ObjectDetector", withExtension: "mlmodelc") else {
            return NSError(domain: "VisionObjectRecognitionViewController", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
        }
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            let objectRecognition = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                DispatchQueue.main.async(execute: {
                    // perform all the UI updates on the main queue
                    if let results = request.results {
                        self.drawVisionRequestResults(results)
                    }
                })
            })
            self.requests = [objectRecognition]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
        
        return error
    }
    
    func drawVisionRequestResults(_ results: [Any]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil // remove all the old recognized objects
        
        var detectionCount = 0
        var colorIndex = 0
        var currentDetections: [String: CGRect] = [:]

        for observation in results where observation is VNRecognizedObjectObservation {
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            
            // Only process detections with confidence > 0.5
            guard objectObservation.confidence > 0.5 else {
                continue
            }
            
            // Select only the label with the highest confidence.
            let topLabelObservation = objectObservation.labels[0]

            // Transform the bounding box from Vision coordinates to UIKit coordinates
            var objectBounds = self.transformBoundingBox(objectObservation.boundingBox)
            
            // Create a unique key for each detection based on identifier and approximate position
            // This allows multiple objects of the same type to be tracked separately
            let centerX = Int(objectBounds.midX / 50) * 50  // Group by 50-pixel regions
            let centerY = Int(objectBounds.midY / 50) * 50
            let detectionKey = "\(topLabelObservation.identifier)_\(centerX)_\(centerY)"
            
            // Apply smoothing to reduce fluctuation, but only for nearby previous detections
            var bestMatch: (key: String, bounds: CGRect, distance: CGFloat)?
            for (prevKey, prevBounds) in previousDetections {
                if prevKey.hasPrefix(topLabelObservation.identifier) {
                    let distance = sqrt(pow(prevBounds.midX - objectBounds.midX, 2) + pow(prevBounds.midY - objectBounds.midY, 2))
                    if distance < 100 { // Only consider matches within 100 pixels
                        if bestMatch == nil || distance < bestMatch!.distance {
                            bestMatch = (prevKey, prevBounds, distance)
                        }
                    }
                }
            }
            
            if let match = bestMatch {
                objectBounds = smoothBounds(current: objectBounds, previous: match.bounds)
            }
            
            currentDetections[detectionKey] = objectBounds

            // Get color for this detection
            let boxColor = boxColors[colorIndex % boxColors.count]
            colorIndex += 1

            let shapeLayer = self.createSharpRectLayerWithBounds(objectBounds, color: boxColor)

            let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                            identifier: topLabelObservation.identifier,
                                                            confidence: topLabelObservation.confidence)
            shapeLayer.addSublayer(textLayer)

            // Add coordinate labels at all four corners
            let coordinateLayers = self.createCoordinateLabels(for: objectBounds, color: boxColor)
            for coordLayer in coordinateLayers {
                shapeLayer.addSublayer(coordLayer)
            }

            detectionOverlay.addSublayer(shapeLayer)
            detectionCount += 1
        }
        
        // Update previous detections for smoothing
        previousDetections = currentDetections
        
        // Update detection count
        self.updateDetectionCount(detectionCount)
        self.updateLayerGeometry()
        CATransaction.commit()
    }
    
    func smoothBounds(current: CGRect, previous: CGRect) -> CGRect {
        // Apply exponential smoothing to reduce jitter
        let x = previous.origin.x * smoothingFactor + current.origin.x * (1 - smoothingFactor)
        let y = previous.origin.y * smoothingFactor + current.origin.y * (1 - smoothingFactor)
        let width = previous.size.width * smoothingFactor + current.size.width * (1 - smoothingFactor)
        let height = previous.size.height * smoothingFactor + current.size.height * (1 - smoothingFactor)
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let exifOrientation = exifOrientationFromDeviceOrientation()
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    override func setupAVCapture() {
        super.setupAVCapture()
        // setup Vision parts
        setupLayers()
        updateLayerGeometry()
        setupVision()
        // Add orientation change observer
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        // start the capture on a background thread to avoid UI unresponsiveness
        DispatchQueue.global(qos: .userInitiated).async {
            self.startCaptureSession()
        }
    }
    
    @objc func orientationDidChange() {
        DispatchQueue.main.async {
            self.updateLayerGeometry()
        }
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    func setupLayers() {
        detectionOverlay = CALayer() // container layer that has all the renderings of the observations
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionOverlay)
        
        // Setup detection count label
        setupDetectionCountLabel()
    }
    
    func updateLayerGeometry() {
        let bounds = rootLayer.bounds
        
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        // Reset transform
        detectionOverlay.setAffineTransform(.identity)
        
        // Set the detection overlay to match the buffer size
        detectionOverlay.bounds = CGRect(x: 0, y: 0, width: bufferSize.width, height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY)
        
        // Calculate scale to match resizeAspectFill behavior of the preview layer
        let scaleX = bounds.size.width / bufferSize.width
        let scaleY = bounds.size.height / bufferSize.height
        let scale = max(scaleX, scaleY) // Use max for aspectFill behavior
        
        // Apply scaling to match the preview layer
        detectionOverlay.setAffineTransform(CGAffineTransform(scaleX: scale, y: scale))
        
        // Update detection count label position for current orientation
        if detectionCountLabel != nil {
            updateDetectionCountPosition()
        }
        CATransaction.commit()
    }
    
    func updateDetectionCountPosition() {
        let bounds = rootLayer.bounds
        let isLandscape = bounds.width > bounds.height
        
        let safeAreaInsets = view.safeAreaInsets
        let padding: CGFloat = 15
        
        if isLandscape {
            // In landscape, position at top-left with safe area consideration
            detectionCountLabel.position = CGPoint(
                x: safeAreaInsets.left + detectionCountLabel.bounds.width / 2 + padding,
                y: safeAreaInsets.top + detectionCountLabel.bounds.height / 2 + padding
            )
        } else {
            // In portrait, position at top-left with safe area consideration
            detectionCountLabel.position = CGPoint(
                x: detectionCountLabel.bounds.width / 2 + padding,
                y: safeAreaInsets.top + detectionCountLabel.bounds.height / 2 + padding
            )
        }
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\nConfidence: %.2f", confidence))
        let largeFont = UIFont(name: "Helvetica-Bold", size: 14.0)!
        formattedString.addAttributes([
            NSAttributedString.Key.font: largeFont,
            NSAttributedString.Key.foregroundColor: UIColor.white
        ], range: NSRange(location: 0, length: formattedString.length))
        
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.width - 10, height: 40)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.minY - 22) // Position above the bounding box
        textLayer.shadowOpacity = 0.8
        textLayer.shadowOffset = CGSize(width: 1, height: 1)
        textLayer.shadowRadius = 2
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.8).cgColor
        textLayer.cornerRadius = 6
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.alignmentMode = .center
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
    func createSharpRectLayerWithBounds(_ bounds: CGRect, color: CGColor) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.borderColor = color
        shapeLayer.borderWidth = 2.0  // Reduced from 3.0 for less prominence
        shapeLayer.backgroundColor = UIColor.clear.cgColor
        shapeLayer.cornerRadius = 0 // Sharp corners
        return shapeLayer
    }
    
    func setupDetectionCountLabel() {
        detectionCountLabel = CATextLayer()
        detectionCountLabel.name = "Detection Count"
        detectionCountLabel.fontSize = 14
        detectionCountLabel.foregroundColor = UIColor.white.cgColor
        detectionCountLabel.backgroundColor = UIColor.black.withAlphaComponent(0.8).cgColor
        detectionCountLabel.cornerRadius = 8
        detectionCountLabel.contentsScale = UIScreen.main.scale
        detectionCountLabel.shadowOpacity = 0.5
        detectionCountLabel.shadowOffset = CGSize(width: 1, height: 1)
        detectionCountLabel.shadowRadius = 2
        detectionCountLabel.alignmentMode = .center
        rootLayer.addSublayer(detectionCountLabel)
    }
    
    func updateDetectionCount(_ count: Int) {
        detectionCountLabel.string = "Total Detections: \(count)"
        
        // Calculate size based on text content
        let textSize = ("Total Detections: \(count)" as NSString).size(withAttributes: [
            NSAttributedString.Key.font: UIFont.systemFont(ofSize: 14)
        ])
        
        detectionCountLabel.bounds = CGRect(x: 0, y: 0, width: textSize.width + 14, height: textSize.height + 6)
        
        // Update position using the orientation-aware function
        updateDetectionCountPosition()
    }
    
    func createCoordinateLabels(for bounds: CGRect, color: CGColor) -> [CATextLayer] {
        var labels: [CATextLayer] = []
        let x = Int(bounds.origin.x)
        let y = Int(bounds.origin.y)
        let w = Int(bounds.origin.x + bounds.size.width)
        let h = Int(bounds.origin.y + bounds.size.height)
        
        // Each corner: (label text, x, y, color)
        let corners: [(String, CGFloat, CGFloat, CGColor)] = [
            ("(\(x),\(y))", bounds.minX, bounds.minY, CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0)), // Top-left (x,y) red
            ("(\(w),\(y))", bounds.maxX, bounds.minY, color), // Top-right (w,y) box color
            ("(\(x),\(h))", bounds.minX, bounds.maxY, color), // Bottom-left (x,h) box color
            ("(\(w),\(h))", bounds.maxX, bounds.maxY, CGColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0)) // Bottom-right (w,h) green
        ]
        
        for (text, px, py, labelColor) in corners {
            let label = CATextLayer()
            label.fontSize = 10
            label.foregroundColor = UIColor.white.cgColor
            label.backgroundColor = UIColor.black.withAlphaComponent(0.8).cgColor
            label.cornerRadius = 3
            label.contentsScale = UIScreen.main.scale
            label.string = text
            
            let textSize = (text as NSString).size(withAttributes: [
                NSAttributedString.Key.font: UIFont.systemFont(ofSize: 10)
            ])
            label.bounds = CGRect(x: 0, y: 0, width: textSize.width + 6, height: textSize.height + 3)
            
            // Place label at the exact corner, with a small offset for readability
            let offset: CGFloat = 4
            var labelPosition = CGPoint(x: px, y: py)
            
            // Offset away from the box for visibility
            if px == bounds.minX { 
                labelPosition.x -= label.bounds.width/2 + offset 
            } else { 
                labelPosition.x += label.bounds.width/2 + offset 
            }
            if py == bounds.minY { 
                labelPosition.y -= label.bounds.height/2 + offset 
            } else { 
                labelPosition.y += label.bounds.height/2 + offset 
            }
            
            label.position = labelPosition
            labels.append(label)
        }
        return labels
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        
        coordinator.animate(alongsideTransition: { _ in
            self.updateLayerGeometry()
        }, completion: nil)
    }
    
    func transformBoundingBox(_ boundingBox: CGRect) -> CGRect {
        // Vision uses normalized coordinates with (0,0) at bottom-left
        // UIKit uses coordinates with (0,0) at top-left
        // We need to convert from Vision's coordinate system to our display coordinate system
        
        // Vision provides normalized coordinates (0-1), convert to buffer coordinates
        // but keep in mind that Vision's Y coordinate system is flipped
        let x = boundingBox.origin.x * bufferSize.width
        let width = boundingBox.size.width * bufferSize.width
        let height = boundingBox.size.height * bufferSize.height
        
        // Flip Y coordinate: Vision has (0,0) at bottom-left, UIKit has (0,0) at top-left
        let y = (1.0 - boundingBox.origin.y - boundingBox.size.height) * bufferSize.height
        
        return CGRect(
            x: x,
            y: y,
            width: width,
            height: height
        )
    }
}
