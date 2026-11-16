"""Start SVP on a private loopback socket owned continuously by this child."""
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    print(f"DMC_SVP_PORT={listener.getsockname()[1]}", flush=True)
    import uvicorn

    uvicorn.Server(uvicorn.Config("app:app", host="127.0.0.1", port=0)).run(
        sockets=[listener]
    )
