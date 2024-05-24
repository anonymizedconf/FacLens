# FacLens Demo
This is the source code of FacLens Demo. You can use the code to deploy FacLens locally.

## Source Code Overview
The source code is structured as follows:

* `ckpt/` contains the checkpoints of fine-tuned FacLens.

* `networks/` includes the code for implementing the model architecture of FacLens.

* `static/` and `templates/` contain files for the demo website.

## Running the Code
To run the code, execute the following command in your terminal:

```bash
$ python app_faclens.py
```

After running the code, open your web browser and enter the following URL to access the demo page: `http://ip_address:10001/demo`. Make sure to replace `ip_address` with your actual IP address.