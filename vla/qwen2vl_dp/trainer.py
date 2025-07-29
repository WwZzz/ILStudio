import transformers.trainer 

class Trainer(transformers.trainer.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        logging_steps = self.args.logging_steps
        if (self.state.global_step % logging_steps == 0) and (self.state.global_step != 0):
            log_dict = {}
            if "action_loss" in outputs:
                log_dict["action_loss"] = outputs["action_loss"].detach().cpu().item()
            if "llm_loss" in outputs:
                log_dict["llm_loss"] = outputs["llm_loss"].detach().cpu().item()
            if log_dict:
                self.log(log_dict)
        return (loss, outputs) if return_outputs else loss