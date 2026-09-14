"""Keep an SSH tunnel to the existing EC2 PostgreSQL container open."""

import subprocess
from pathlib import Path

from dotenv import dotenv_values


def main():
    env = dotenv_values(Path(__file__).with_name('.env'))
    ssh = [
        'ssh', '-i', env['EC2_KEY_PATH'],
        '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=15',
    ]
    target = f"{env['EC2_USER']}@{env['EC2_HOST']}"
    # Resolve the address each time: Docker may change it after recreation.
    address = subprocess.check_output(
        ssh + [target, 'sudo -n docker inspect chatbot-postgres --format '
               "'{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}'"],
        text=True, timeout=30,
    ).split()
    if len(address) != 1:
        raise RuntimeError('Expected one PostgreSQL container network address.')
    port = int(env.get('CHECKPOINT_TUNNEL_PORT', '15432'))
    print(f'Forwarding 127.0.0.1:{port} to EC2 PostgreSQL. Keep this running.', flush=True)
    subprocess.run(
        ssh + ['-N', '-o', 'ExitOnForwardFailure=yes',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
               '-L', f'127.0.0.1:{port}:{address[0]}:5432', target],
        check=True,
    )


if __name__ == '__main__':
    main()
