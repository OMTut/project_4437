from ultralytics import YOLO
import torch

def main():

   # New model
   model = YOLO('yolo11n.yaml')

   # Load pretrained yolo model
   model = YOLO('yolo11n.pt')

   # Parameter for early stopping
   max_epochs = 2
   patience = 5 # of epochs to stop if no improvement
   best_val_loss = float('inf')
   patience_counter = 0

   for epoch in range(max_epochs):
      print(f"Epoch {epoch+1} of {max_epochs}")

       #Train the model
      # note: add resule=True to resume from previous training that reached max_epochs
      results = model.train(data='data_SWT.yaml', epochs=1, imgsz=640)

      # Evaluate the model
      val_results = model.val()

      # Extract validation box loss
      current_val_loss = val_results.results_dict.get('val/box_loss', None)
      if current_val_loss is None:
         print("Validation box loss not found, Skipping early stopping check")
         continue

      # Log the current loss and best loss
      print(f"Epoch {epoch + 1}: current_val_loss = {current_val_loss}, best_val_loss = {best_val_loss}")

      # Check for improvement
      if current_val_loss < best_val_loss:
         best_val_loss = current_val_loss
         patience_counter = 0
         print(f"New best val_loss: {best_val_loss}. Saving model...")
         model.save('best.pt')
      else:
         patience_counter += 1
         print(f"No improvement in val_loss for {patience_counter} epochs")
      
      # Stop training at end of patience
      if patience_counter >= patience:
         print(f"Early stopping after {epoch+1} epochs")
         break


   # Export the model to ONNX format
   success = model.export(format='onnx')
   if success:
      print("Model exported successfully")

if __name__ == '__main__':
    main()

# @software{yolo11_ultralytics,
#   author = {Glenn Jocher and Jing Qiu},
#   title = {Ultralytics YOLO11},
#   version = {11.0.0},
#   year = {2024},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
# }