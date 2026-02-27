from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class User(models.Model):
    username = models.CharField(verbose_name="用户名", max_length=16, unique=True)
    password = models.CharField(verbose_name="密码", max_length=128)
    qx= models.IntegerField(verbose_name="权限",choices=((0, '普通用户'),(1, '管理员')),default=0)
    def save(self, *args, **kwargs):
        if not self.password.startswith('pbkdf2_'):
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)
    def __str__(self):
        return self.username
    class Meta:
        verbose_name = "用户"
        verbose_name_plural = verbose_name


class Gp(models.Model):
    code = models.CharField(verbose_name="股票代码", max_length=128)
    date= models.DateField(verbose_name="日期")
    open = models.FloatField(verbose_name="开盘价")
    close = models.FloatField(verbose_name="收盘价")
    high = models.FloatField(verbose_name="最高价")
    low = models.FloatField(verbose_name="最低价")
    volume = models.FloatField(verbose_name="成交量")
    def __str__(self):
        return self.code
    class Meta:
        verbose_name = "股票"
        verbose_name_plural = verbose_name