import { Keypair } from "@solana/web3.js";
import fs from "fs";

const keypair = Keypair.generate();
console.log("Public Key:", keypair.publicKey.toBase58());

// Save secret key as Base58 for .env
import bs58 from "bs58";
const base58PrivateKey = bs58.encode(keypair.secretKey);
console.log("Base58 Private Key:", base58PrivateKey);

fs.writeFileSync("devnet-key.json", JSON.stringify({
  publicKey: keypair.publicKey.toBase58(),
  secretKeyBase58: base58PrivateKey
}, null, 2));

console.log("Saved to devnet-key.json. Use secretKeyBase58 in .env");
