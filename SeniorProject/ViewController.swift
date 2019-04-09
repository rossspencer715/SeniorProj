//
//  ViewController.swift
//  Senior Project
//
//  Created by Ross Spencer on 1/29/19.
//  Copyright © 2019 Ross Spencer. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

    @IBOutlet weak var peanutImageView: UIImageView!
    @IBOutlet weak var openCVVersionLabel: UILabel!
    override func viewDidLoad() {
        print("\(OpenCVWrapper.openCVVersionString())")
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        //invoke functions in openCVwrapper class from Swift
        openCVVersionLabel.text = OpenCVWrapper.openCVVersionString()
    }
    
    
    @IBAction func toGreyscaleTouched(_ sender: UIButton) {
        peanutImageView.image = OpenCVWrapper.makeGray(peanutImageView.image!)
    }
    
    @IBAction func showMessage(sender: UIButton) {
        let alertController = UIAlertController(title: "Welcome to My First App", message: "Hello World", preferredStyle: UIAlertController.Style.alert)
        alertController.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
    @IBAction func importImage(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.photoLibrary){
            let image = UIImagePickerController()
            image.delegate = self
            image.sourceType = .photoLibrary
            image.allowsEditing = false
            self.present(image, animated: true){
                //image is imported
            }
        }
        else{
            //ERROR
        }
    }
    
    @IBAction func takePicture(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera){
            let image = UIImagePickerController()
            image.delegate = self
            image.sourceType = .camera
            image.allowsEditing = false
            self.present(image, animated: false, completion: nil)
        }
        else{
            print("oopsies")
            //ERROR
        }
        
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage{
            peanutImageView.image = image
        }
        else{
            print("hmm")
        }
        self.dismiss(animated: true, completion: nil)
    }
    
    @IBOutlet weak var maturityLevel: UILabel!
    @IBAction func classify(_ sender: UIButton) {
        
        print("Something To Print1");
        maturityLevel.text = OpenCVWrapper.classifyPeanut(peanutImageView.image!)
        print("Something To Print2");
    }

}

