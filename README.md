# ByteWAF

ByteWAF is a Web Application Firewall (WAF) designed to provide enhanced security for web applications. It integrates semantic detection models using ONNX Runtime and is built with OpenResty for high-performance Lua scripting.

## Features

- **Semantic Detection**: Uses ONNX Runtime for semantic analysis of requests to detect malicious activities.
- **Custom Rules**: Supports custom Lua-based rules for flexible security policies.
- **IP Blocking**: Implements IP-based blocking using shared dictionaries.
- **High Performance**: Built on OpenResty for efficient request handling.
- **GPU Acceleration**: Leverages NVIDIA GPUs for model inference.

---

## Project Structure

```
bytewaf/
├── docker-compose.yml       # Docker Compose configuration
├── src/                     # WAF service source code
│   ├── lua/
│   │   └── waf.lua          # Core WAF logic
│   └── nginx.conf           # OpenResty configuration
├── script/                  # ONNX service source code
│   ├── app.py               # ONNX model API
│   └── requirements.txt     # Python dependencies
├── model_save/              # Directory for ONNX model
│   └── model.onnx           # Pre-trained ONNX model
└── README.md                # Project documentation
```

---

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for ONNX Runtime GPU acceleration)

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/bytewaf.git
    cd bytewaf
    ```

2. Place your ONNX model in the `model_save/` directory:
    ```
    model_save/model.onnx
    ```

3. Build and start the services using Docker Compose:
    ```bash
    docker-compose up --build
    ```

---

## Configuration

### `docker-compose.yml`

- **WAF Service**:
  - Listens on port `5557`.
  - Communicates with the ONNX service at `http://172.31.0.11:5555/detect`.

- **ONNX Service**:
  - Exposes the model API on port `5555`.
  - Requires GPU access for inference.

### `nginx.conf`

Customize the OpenResty configuration to define routing and logging.

### `lua/waf.lua`

Modify the Lua script to add or update WAF rules.

---

## Usage

1. Access the WAF service at `http://localhost:5557`.
2. The WAF will analyze incoming requests and block malicious traffic based on:
    - Semantic model predictions.
    - Custom Lua rules.

---

## Development

### WAF Service

- Located in the `src/` directory.
- Core logic is implemented in `lua/waf.lua`.

### ONNX Service

- Located in the `script/` directory.
- Python-based API for serving ONNX models.

---

## Example

### Blocked Request

If a request is flagged as malicious, the WAF will respond with:

```json
{
  "code": 510,
  "msg": "Blocked by semantic model: <reason>"
}
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [OpenResty](https://openresty.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)
