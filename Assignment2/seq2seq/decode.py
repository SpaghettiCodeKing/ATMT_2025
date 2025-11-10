import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
from seq2seq.beam import BeamSearch, BeamSearchNode  # assumes you saved your beamsearch classes here


def decode(model: Seq2SeqModel,
           src_tokens: torch.Tensor,
           src_pad_mask: torch.Tensor,
           max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor,
           args,
           device: torch.device):
    """
    Decodes sequences using greedy or beam search, depending on args.beam_size.
    If beam_size == 1 → greedy decoding (default)
    If beam_size > 1 → beam search decoding
    """
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    batch_size = src_tokens.size(0)
    beam_size = getattr(args, "beam_size", 1)

    # Run encoder once for efficiency
    encoder_out = model.encoder(src_tokens, src_pad_mask)

    # ==============================================================
    # =============== GREEDY DECODING (beam_size == 1) =============
    # ==============================================================
    if beam_size == 1:
        generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_out_len):
            max_len = model.decoder.pos_embed.size(1)
            if generated.size(1) > max_len:
                generated = generated[:, :max_len]

            trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)

            output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
    
            output = model.decoder(encoder_out, src_pad_mask, generated, trg_pad_mask)
            next_token_logits = output[:, -1, :]  # last time step
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == EOS)
            if finished.all():
                break

        predicted_tokens = []
        for seq in generated[:, 1:].tolist():
            if EOS in seq:
                seq = seq[: seq.index(EOS) + 1]
            predicted_tokens.append(seq)
        return predicted_tokens

    # ==============================================================
    # =============== BEAM SEARCH DECODING (beam_size > 1) =========
    # ==============================================================
    else:
        results = []
        for b in range(batch_size):
            beam = BeamSearch(beam_size=beam_size, max_len=max_out_len, pad=PAD)

            init_seq = torch.tensor([BOS], dtype=torch.long, device=device)
            init_node = BeamSearchNode(
                search=beam,
                emb=None, lstm_out=None, final_hidden=None, final_cell=None, mask=None,
                sequence=init_seq,
                logProb=0.0,
                length=1
            )
            beam.add(-init_node.eval(), init_node)

            for _ in range(max_out_len):
                current_nodes = beam.get_current_beams()
                if not current_nodes:
                    break

                for score, node in current_nodes:
                    seq = node.sequence.unsqueeze(0)
                    trg_pad_mask = (seq == PAD).unsqueeze(1).unsqueeze(2)

                    logits = model.decoder(encoder_out[b:b+1], src_pad_mask[b:b+1], seq, trg_pad_mask)
                    next_logits = logits[:, -1, :]
                    probs = torch.log_softmax(next_logits, dim=-1).squeeze(0)
                    topk_logprobs, topk_ids = torch.topk(probs, beam_size)

                    for k in range(beam_size):
                        token_id = topk_ids[k].item()
                        logp = node.logp + topk_logprobs[k].item()
                        new_seq = torch.cat([node.sequence, torch.tensor([token_id], device=device)])
                        new_node = BeamSearchNode(beam, None, None, None, None, None, new_seq, logp, node.length + 1)

                        if token_id == EOS:
                            beam.add_final(-new_node.eval(), new_node)
                        else:
                            beam.add(-new_node.eval(), new_node)

                beam.prune()

            best_score, best_node = beam.get_best()
            results.append(best_node.sequence.tolist())

        return results
